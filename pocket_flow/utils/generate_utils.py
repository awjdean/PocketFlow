"""
Utilities for ligand graph growth and RDKit molecule reconstruction.

This module contains helper functions used during generation to:

- Incrementally append atoms/bonds to a `torch_geometric.data.Data` object that
  represents the ligand "context" graph used by the model.
- Convert a generated ligand context graph back into an RDKit molecule with 3D
  coordinates.
- Apply a few chemistry-specific postprocessing rules (e.g., preventing small
  triangles, reducing overly aggressive double bonds, and stabilizing certain
  charged substructures).

Conventions used throughout:

- Element identities are typically represented by atomic numbers (e.g., 6 for C),
  but `add_ligand_atom_to_data` expects `element` to be an *index* into
  `ELEMENT_TYPE_MAP` (unless you pass a custom `type_map`).
- Bond orders are encoded as integers:
  `1` (single), `2` (double), `3` (triple), and `12` (aromatic).
- The `Data` object is expected to carry several ligand context fields, such as:
  `ligand_context_pos`, `ligand_context_element`, `ligand_context_bond_index`,
  `ligand_context_bond_type`, and `ligand_context_feature_full`.
"""

from __future__ import annotations

import copy
import itertools
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, Geometry
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import Mol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch_geometric.data import Data

MAX_VALENCE_DICT: dict[int, torch.Tensor] = {
    6: torch.LongTensor([4]),
    7: torch.LongTensor([3]),
    8: torch.LongTensor([2]),
    9: torch.LongTensor([1]),
    15: torch.LongTensor([5]),
    16: torch.LongTensor([6]),
    17: torch.LongTensor([1]),
    35: torch.LongTensor([1]),
    53: torch.LongTensor([1]),
    1: torch.LongTensor([1]),
}

ELEMENT_TYPE_MAP: tuple[int, ...] = (1, 6, 7, 8, 9, 15, 16, 17, 35, 53)


def add_ligand_atom_to_data(
    old_data: Data,
    pos: torch.Tensor,
    element: torch.Tensor,
    bond_index: torch.Tensor,
    bond_type: torch.Tensor,
    type_map: tuple[int, ...] = ELEMENT_TYPE_MAP,
    max_valence_dict: dict[int, torch.Tensor] = MAX_VALENCE_DICT,
) -> Data:
    """
    Append a new ligand atom (and optional bonds) to a ligand context `Data`.

    The input `old_data` is cloned and the returned `Data` contains:

    - Updated `ligand_context_pos` with the new atom's 3D position.
    - Updated `ligand_context_feature_full` with a one-hot element feature plus
      bookkeeping features (neighbor count, valence, and bond-count buckets).
    - Updated `ligand_context_element`, `ligand_context_bond_index`,
      `ligand_context_bond_type`, `ligand_context_valence`, and `max_atom_valence`.

    Notes:
        - `element` is interpreted as an index into `type_map` (not an atomic
          number). The stored `ligand_context_element` uses atomic numbers.
        - If `bond_type` is non-empty, the helper applies heuristics to avoid
          forming triangles and to reduce bond orders that would likely violate
          simple valence rules.

    Args:
        old_data: Original ligand context graph. It is not mutated.
        pos: Position of the new atom, shape `(3,)`.
        element: Element *index* into `type_map`, scalar tensor.
        bond_index: Candidate edges from the new atom to existing atoms,
            shape `(2, E)` (the first row is overwritten to point to the new atom).
        bond_type: Candidate bond orders for `bond_index`, shape `(E,)`.
        type_map: Mapping from model element indices to atomic numbers.
        max_valence_dict: Mapping from atomic number to a 1D tensor containing
            that element's maximum allowed valence.

    Returns:
        A cloned `Data` with the new atom/bonds incorporated into the ligand context.
    """
    data = old_data.clone()
    if "max_atom_valence" not in data.__dict__["_store"]:
        data.max_atom_valence = torch.empty(0, dtype=torch.long)
    # add position of new atom to context
    data.ligand_context_pos = torch.cat(
        [data.ligand_context_pos, pos.view(1, 3).to(data.ligand_context_pos)], dim=0
    )

    # add feature of new atom to context
    data.ligand_context_feature_full = torch.cat(
        [
            data.ligand_context_feature_full,
            torch.cat(
                [
                    F.one_hot(element.view(1), len(type_map)).to(
                        data.ligand_context_feature_full
                    ),  # (1, num_elements)
                    torch.tensor([[1, 0, 0]]).to(
                        data.ligand_context_feature_full
                    ),  # is_mol_atom, num_neigh (placeholder), valence (placeholder)
                    torch.tensor([[0, 0, 0]]).to(
                        data.ligand_context_feature_full
                    ),  # num_of_bonds 1, 2, 3(placeholder)
                ],
                dim=1,
            ),
        ],
        dim=0,
    )
    data.context_idx = torch.arange(data.context_idx.size(0) + 1)
    #
    idx_num_neigh = len(type_map) + 1
    idx_valence = idx_num_neigh + 1
    idx_num_of_bonds = idx_valence + 1

    # add type of new atom to context
    element_idx = int(element.item())
    element = torch.LongTensor([type_map[element_idx]])
    data.ligand_context_element = torch.cat(
        [data.ligand_context_element, element.view(1).to(data.ligand_context_element)]
    )
    element_type = int(element.item())
    max_new_atom_valence = max_valence_dict[element_type].to(data.max_atom_valence)
    data.max_atom_valence = torch.cat([data.max_atom_valence, max_new_atom_valence])

    # change the feature of new atom to context according to ligand context
    if len(bond_type) != 0:
        bond_index, bond_type = remove_triangle(
            pos,
            data.ligand_context_pos,
            data.ligand_context_bond_index,
            data.ligand_context_bond_type,
            bond_index,
            bond_type,
        )
        bond_index[0, :] = len(data.ligand_context_pos) - 1
        bond_type = check_valence_is_2(
            bond_index, bond_type, data.ligand_context_element, data.ligand_context_valence
        )
        bond_vec = data.ligand_context_pos[bond_index[0]] - data.ligand_context_pos[bond_index[1]]
        bond_lengths = torch.norm(bond_vec, dim=-1, p=2)
        if (bond_lengths > 3).any():
            print(bond_lengths)

        bond_index_all = torch.cat(
            [bond_index, torch.stack([bond_index[1, :], bond_index[0, :]], dim=0)], dim=1
        )
        bond_type_all = torch.cat([bond_type, bond_type], dim=0)

        data.ligand_context_bond_index = torch.cat(
            [data.ligand_context_bond_index, bond_index_all.to(data.ligand_context_bond_index)], dim=1
        )

        data.ligand_context_bond_type = torch.cat([data.ligand_context_bond_type, bond_type_all])
        # modify atom features related to bonds
        # previous atom
        data.ligand_context_feature_full[bond_index[1, :], idx_num_neigh] += (
            1  # num of neigh of previous nodes
        )
        data.ligand_context_feature_full[bond_index[1, :], idx_valence] += (
            bond_type  # valence of previous nodes
        )
        data.ligand_context_feature_full[bond_index[1, :], idx_num_of_bonds + bond_type - 1] += (
            1  # num of bonds of
        )
        # the new atom
        data.ligand_context_feature_full[-1, idx_num_neigh] += len(bond_index[1])  # num of neigh of last node
        data.ligand_context_feature_full[-1, idx_valence] += torch.sum(bond_type)  # valence of last node
        for bond in [1, 2, 3]:
            data.ligand_context_feature_full[-1, idx_num_of_bonds + bond - 1] += (
                bond_type == bond
            ).sum()  # num of bonds of last node
    data.ligand_context_valence = data.ligand_context_feature_full[:, idx_valence]
    del old_data
    return data


def data2mol(data: Data, raise_error: bool = True, sanitize: bool = True) -> Mol:
    """
    Reconstruct an RDKit molecule from a ligand context graph.

    The function reads ligand context attributes from `data`, creates an RDKit
    `RWMol` with a 3D conformer, adds bonds according to integer bond types, and
    then performs light postprocessing:

    - `modify_submol` to assign charges for a specific substructure.
    - A SMILES roundtrip to validate basic chemistry.
    - Optional sanitization (with Kekulization/aromaticity handling).

    Args:
        data: A `torch_geometric.data.Data` object with ligand context fields:
            `ligand_context_element`, `ligand_context_bond_index`,
            `ligand_context_bond_type`, and `ligand_context_pos`.
        raise_error: If True, raise `MolReconsError` on reconstruction failure;
            otherwise print a message and return the best-effort molecule.
        sanitize: If True, run `Chem.SanitizeMol` with a relaxed sanitization mask
            (Kekulization/aromaticity are handled separately).

    Returns:
        An RDKit `Mol` with 3D coordinates.

    Raises:
        MolReconsError: If the reconstructed molecule fails a SMILES roundtrip and
            `raise_error=True`.
        ValueError: If an unknown bond order integer is encountered.
    """
    element: list[int] = data.ligand_context_element.clone().cpu().tolist()
    bond_index: list[list[int]] = data.ligand_context_bond_index.clone().cpu().tolist()
    bond_type: list[int] = data.ligand_context_bond_type.clone().cpu().tolist()
    pos: list[list[float]] = data.ligand_context_pos.clone().cpu().tolist()
    n_atoms = len(pos)
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)

    # add atoms and coordinates
    for i, atom in enumerate(element):
        rd_atom = Chem.Atom(atom)
        rd_mol.AddAtom(rd_atom)
        rd_coords = Geometry.Point3D(*pos[i])
        rd_conf.SetAtomPosition(i, rd_coords)
    rd_mol.AddConformer(rd_conf)
    # add bonds
    for i, type_this in enumerate(bond_type):
        node_i, node_j = bond_index[0][i], bond_index[1][i]
        if node_i < node_j:
            if type_this == 1:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.SINGLE)
            elif type_this == 2:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.DOUBLE)
            elif type_this == 3:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.TRIPLE)
            elif type_this == 12:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.AROMATIC)
            else:
                raise ValueError(f"unknown bond order {type_this}")
    # modify
    if raise_error:
        rd_mol = modify_submol(rd_mol)
    else:
        try:
            rd_mol = modify_submol(rd_mol)
        except Exception:
            print("MolReconsError")
    # check valid
    rd_mol_check = Chem.MolFromSmiles(Chem.MolToSmiles(rd_mol))
    if rd_mol_check is None:
        if raise_error:
            raise MolReconsError()
        else:
            print("MolReconsError")
    rd_mol = rd_mol.GetMol()
    if 12 in bond_type:  # mol may directly come from true mols and contains aromatic bonds
        Chem.Kekulize(rd_mol, clearAromaticFlags=True)
    if sanitize:
        Chem.SanitizeMol(rd_mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_SETAROMATICITY)
    return rd_mol


def modify_submol(mol: Chem.RWMol) -> Chem.RWMol:
    """
    Assign formal charges for the substructure `C=N(C)O`.

    RDKit sanitization can fail or assign undesired valences for some generated
    fragments. This helper detects occurrences of the SMARTS `C=N(C)O` (without
    sanitizing) and sets charges to the zwitterionic form:

    - N becomes `+1`
    - O becomes `-1`

    Args:
        mol: Editable RDKit molecule.

    Returns:
        The same `RWMol` with formal charges updated in-place.
    """
    submol = Chem.MolFromSmiles("C=N(C)O", sanitize=False)
    sub_fragments = mol.GetSubstructMatches(submol)
    for fragment in sub_fragments:
        atomic_nums = np.array([mol.GetAtomWithIdx(atom).GetAtomicNum() for atom in fragment])
        idx_atom_N = fragment[np.where(atomic_nums == 7)[0][0]]
        idx_atom_O = fragment[np.where(atomic_nums == 8)[0][0]]
        mol.GetAtomWithIdx(idx_atom_N).SetFormalCharge(1)  # set N to N+
        mol.GetAtomWithIdx(idx_atom_O).SetFormalCharge(-1)  # set O to O-
    return mol


class MolReconsError(Exception):
    """Raised when an RDKit molecule cannot be reconstructed from a graph."""

    pass


def add_context(data: Data) -> Data:
    """
    Populate ligand context fields from ligand fields.

    This is a convenience function used when the initial "context" is the full
    ligand graph (e.g., for evaluation or for bootstrapping generation).

    Args:
        data: A `Data` object with ligand fields: `ligand_pos`, `ligand_element`,
            `ligand_bond_index`, and `ligand_bond_type`.

    Returns:
        The same `Data` object with `ligand_context_*` fields set.
    """
    data.ligand_context_pos = data.ligand_pos
    data.ligand_context_element = data.ligand_element
    data.ligand_context_bond_index = data.ligand_bond_index
    data.ligand_context_bond_type = data.ligand_bond_type
    return data


def check_valency(mol: Mol) -> bool:
    """
    Check whether an RDKit molecule has acceptable valence properties.

    This runs a limited RDKit sanitization (`SANITIZE_PROPERTIES`), which is
    commonly used as a quick valency/chemistry sanity check without fully
    rewriting aromaticity or Kekulization.

    Args:
        mol: RDKit molecule to check.

    Returns:
        True if the sanitization step succeeds, otherwise False.
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except ValueError:
        return False


def check_double_bond(
    ligand_context_bond_index: torch.Tensor,
    ligand_context_bond_type: torch.Tensor,
    bond_index_to_add: torch.Tensor,
    bond_type_to_add: torch.Tensor,
) -> torch.Tensor:
    """
    Reduce proposed bond orders when the target atom already has a double bond.

    For each destination atom in `bond_index_to_add[1]`, this checks whether the
    existing context already contains at least one double bond incident to that
    atom (as represented in `ligand_context_bond_index`/`bond_type`). If so, and
    the proposed bond order is >= 2, the proposed bond order is reduced by 1.

    This is a heuristic intended to lower the rate of obvious over-valence in
    generated molecules.

    Args:
        ligand_context_bond_index: Existing context edges, shape `(2, E_ctx)`.
        ligand_context_bond_type: Existing context bond orders, shape `(E_ctx,)`.
        bond_index_to_add: Proposed edges to add, shape `(2, E_new)`.
        bond_type_to_add: Proposed bond orders for `bond_index_to_add`, shape `(E_new,)`.

    Returns:
        Potentially adjusted `bond_type_to_add`, shape `(E_new,)`.
    """
    bond_type_in_place = [
        ligand_context_bond_type[ix == ligand_context_bond_index[0]] for ix in bond_index_to_add[1]
    ]
    has_double = torch.BoolTensor([(bt == 2).sum() > 0 for bt in bond_type_in_place])
    bond_type_to_add_has_double = bond_type_to_add >= 2
    mask = torch.stack([has_double, bond_type_to_add_has_double]).all(dim=0)
    bond_type_to_add = bond_type_to_add - mask.long()
    return bond_type_to_add


def check_valence_is_2(
    bond_index_to_add: torch.Tensor,
    bond_type_to_add: torch.Tensor,
    ligand_context_element: torch.Tensor,
    ligand_context_valence: torch.Tensor,
) -> torch.Tensor:
    """
    Reduce proposed bond orders when connecting carbon-to-carbon at high valence.

    If the newly added atom (last element in `ligand_context_element`) is carbon,
    then for each proposed neighbor:

    - If the neighbor is also carbon,
    - and the neighbor already has valence >= 2 (according to `ligand_context_valence`),
    - and the proposed bond order is >= 2,

    then the proposed bond order is reduced by 1.

    This is a targeted heuristic to avoid building unrealistic C=C or C#C bonds
    onto already "saturated" carbon atoms during incremental generation.

    Args:
        bond_index_to_add: Proposed edges to add, shape `(2, E_new)`.
        bond_type_to_add: Proposed bond orders, shape `(E_new,)`.
        ligand_context_element: Context atomic numbers, shape `(N_ctx,)`.
        ligand_context_valence: Context valence accumulator, shape `(N_ctx,)`.

    Returns:
        Potentially adjusted `bond_type_to_add`, shape `(E_new,)`.
    """
    atom_type_to_add = ligand_context_element[-1]
    new_atom_type_is_C = (atom_type_to_add == 6).view(-1)
    if new_atom_type_is_C:
        atom_valence_in_place = ligand_context_valence[bond_index_to_add[1]] >= 2
        atom_type_in_place_is_C = ligand_context_element[bond_index_to_add[1]] == 6
        bond_type_to_add_mask = bond_type_to_add >= 2

        mask = torch.stack([atom_valence_in_place, atom_type_in_place_is_C, bond_type_to_add_mask]).all(dim=0)
        bond_type_to_add = bond_type_to_add - mask.long()
    return bond_type_to_add


def remove_triangle(
    pos_to_add: torch.Tensor,
    ligand_context_pos: torch.Tensor,
    ligand_context_bond_index: torch.Tensor,
    _ligand_context_bond_type: torch.Tensor,
    bond_index_to_add: torch.Tensor,
    bond_type_to_add: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prevent adding multiple bonds that would create a short triangle.

    When the new atom is connected to multiple existing atoms, it is possible
    that two (or more) of those target atoms are already connected to each
    other in the existing context. Adding both bonds would then form a
    three-membered ring (triangle), which is rarely chemically plausible for
    typical drug-like ligands.

    This heuristic detects that situation and removes *one* of the proposed
    bonds: it drops the bond whose target atom is farthest from the new atom
    (based on Euclidean distance in `ligand_context_pos`).

    Args:
        pos_to_add: Position of the new atom, shape `(3,)`.
        ligand_context_pos: Positions of existing context atoms, shape `(N_ctx, 3)`.
        ligand_context_bond_index: Existing context edges, shape `(2, E_ctx)`.
        _ligand_context_bond_type: Unused (kept for call-site signature symmetry).
        bond_index_to_add: Proposed edges to add, shape `(2, E_new)`.
        bond_type_to_add: Proposed bond orders, shape `(E_new,)`.

    Returns:
        A tuple `(bond_index_to_add, bond_type_to_add)` with at most one edge removed.
    """
    new_j = bond_index_to_add[1]
    atom_in_place_adjs = [ligand_context_bond_index[1][ligand_context_bond_index[0] == i] for i in new_j]
    L: list[list[bool]] = []
    for j in new_j:
        matches: list[bool] = []
        for i in atom_in_place_adjs:
            if j in i:
                matches.append(True)
            else:
                matches.append(False)
        L.append(matches)
    adj_mask = torch.LongTensor(L).any(-1)
    if adj_mask.sum() > 0:
        dist = torch.norm(pos_to_add.view(-1, 3) - ligand_context_pos[new_j], dim=-1, p=2)
        dist_mask_idx = torch.nonzero(adj_mask).view(-1)
        max_dist_idx_to_remove = dist_mask_idx[dist[dist_mask_idx].argmax()]
        mask = torch.arange(len(bond_type_to_add), device=bond_type_to_add.device) != max_dist_idx_to_remove
        bond_index_to_add = bond_index_to_add[:, mask]
        bond_type_to_add = bond_type_to_add[mask]
    return bond_index_to_add, bond_type_to_add


PATTERNS: list[Mol | None] = [
    Chem.MolFromSmarts("[N]1~&@[N]~&@[C]~&@[C]~&@[C]~&@[C]~&@1"),
    Chem.MolFromSmarts("[N]1~&@[C]~&@[N]~&@[C]~&@[C]~&@[C]~&@1"),
    Chem.MolFromSmarts("[N]1~&@[C]~&@[C]~&@[N]~&@[C]~&@[C]~&@1"),
    Chem.MolFromSmarts("[N]1~&@[C]~&@[C]~&@[C]~&@[C]~&@[C]~&@1"),
    Chem.MolFromSmarts("[C]1~&@[C]~&@[C]~&@[C]~&@[C]~&@[C]~&@1"),
]

PATTERNS_1: list[list[Mol | None]] = [
    [
        Chem.MolFromSmarts("[#6,#7,#8]-[#7]1~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@1"),
        Chem.MolFromSmarts("[C,N,O]-[N]1~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@1"),
    ],
    [
        Chem.MolFromSmarts(
            "[#6,#7,#8]-[#6]1(-[#6,#7,#8])~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@1"
        ),
        Chem.MolFromSmarts("[C,N,O]-[C]1(-[C,N,O])~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@1"),
    ],
]

MAX_VALENCE: dict[str, int] = {"C": 4, "N": 3}


def modify(mol: Mol, max_double_in_6ring: int = 0) -> Mol:
    """
    Apply bond-order/aromaticity heuristics to stabilize generated molecules.

    The generation process can produce chemically awkward assignments of double
    bonds/aromaticity in six-membered rings. This function applies a series of
    SMARTS-based rules to:

    - Limit the number of explicit double bonds in 6-member rings
      (`max_double_in_6ring`).
    - Convert certain non-ring double bonds to single bonds in specific motifs.
    - Mark some rings as aromatic when they contain too many double bonds and
      none of the atoms are fully saturated SP3 at max valence.

    The function preserves 3D coordinates by rebuilding an output molecule from
    the modified `RWMol` and copying its conformer positions.

    Args:
        mol: Input RDKit molecule.
        max_double_in_6ring: Maximum number of explicit double bonds allowed in a
            six-membered ring before heuristics reduce/convert bond orders.

    Returns:
        A potentially modified RDKit `Mol`. If postprocessing produces an invalid
        molecule (SMILES roundtrip or Kekulization fails), the original molecule
        is returned.
    """
    mol_copy = copy.deepcopy(mol)
    mw = Chem.RWMol(mol)

    p1 = Chem.MolFromSmarts("[#6,#7]=[#6]1-[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]-1")
    p1_ = Chem.MolFromSmarts("[C,N]=[C]1-[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]-1")
    subs = set(list(mw.GetSubstructMatches(p1)) + list(mw.GetSubstructMatches(p1_)))
    for sub in subs:
        comb = itertools.combinations(sub, 2)
        change_double = False
        r_b_double = 0
        b_list: list[tuple[tuple[int, int], str, bool]] = []
        for ix, c in enumerate(comb):
            b = mw.GetBondBetweenAtoms(*c)
            if ix == 0:
                bt = b.GetBondType().__str__()
                is_r = b.IsInRingSize(6)
                b_list.append((c, bt, is_r))
                continue
            if b is not None:
                bt = b.GetBondType().__str__()
                is_r = b.IsInRing()
                b_list.append((c, bt, is_r))
                if is_r is True and bt == "DOUBLE":
                    r_b_double += 1
                    if r_b_double > max_double_in_6ring:
                        change_double = True
        if change_double:
            for ix, b in enumerate(b_list):
                if ix == 0:
                    if b[-1] is False:
                        mw.RemoveBond(*b[0])
                        mw.AddBond(*b[0], Chem.rdchem.BondType.SINGLE)
                    else:
                        continue
                if b[1] == "DOUBLE" and b[-1] is False:
                    mw.RemoveBond(*b[0])
                    mw.AddBond(*b[0], Chem.rdchem.BondType.SINGLE)
                    break

    for p2 in PATTERNS_1:
        Chem.GetSSSR(mw)
        subs2 = set(list(mw.GetSubstructMatches(p2[0])) + list(mw.GetSubstructMatches(p2[1])))
        for sub in subs2:
            comb = itertools.combinations(sub, 2)
            b_list_2 = [
                (c, mw.GetBondBetweenAtoms(*c)) for c in comb if mw.GetBondBetweenAtoms(*c) is not None
            ]
            for b in b_list_2:
                if b[-1].GetBondType().__str__() == "DOUBLE":
                    mw.RemoveBond(*b[0])
                    mw.AddBond(*b[0], Chem.rdchem.BondType.SINGLE)

    Chem.GetSSSR(mw)
    p3 = Chem.MolFromSmarts("[#8]=[#6]1-[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]-1")
    p3_ = Chem.MolFromSmarts("[O]=[C]1-[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]-1")
    subs = set(list(mw.GetSubstructMatches(p3)) + list(mw.GetSubstructMatches(p3_)))
    subs_set_2: list[set[int]] = [set(s) for s in subs]
    for sub in subs:
        comb = itertools.combinations(sub, 2)
        b_list_3 = [(c, mw.GetBondBetweenAtoms(*c)) for c in comb if mw.GetBondBetweenAtoms(*c) is not None]
        for b in b_list_3:
            if b[-1].GetBondType().__str__() == "DOUBLE" and b[-1].IsInRing() is True:
                mw.RemoveBond(*b[0])
                mw.AddBond(*b[0], Chem.rdchem.BondType.SINGLE)

    p = Chem.MolFromSmarts("[#6,#7]1~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@1")
    p_ = Chem.MolFromSmarts("[C,N]1~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@1")
    Chem.GetSSSR(mw)
    subs = set(list(mw.GetSubstructMatches(p)) + list(mw.GetSubstructMatches(p_)))
    for sub in subs:
        pass_sub = False
        if subs_set_2:
            for s in subs_set_2:
                if len(s - set(sub)) == 1:
                    pass_sub = True
                    break
        if pass_sub:
            continue

        bond_list: list[tuple[int, int]] = [
            (i, sub[0]) if ix + 1 == len(sub) else (i, sub[ix + 1]) for ix, i in enumerate(sub)
        ]
        if len(bond_list) == 0:
            continue
        atoms = [mw.GetAtomWithIdx(i) for i in sub]
        for a in atoms:
            if (
                a.GetExplicitValence() == MAX_VALENCE[a.GetSymbol()]
                and a.GetHybridization().__str__() == "SP3"
            ):
                break
        else:
            bond_type_strs = [mw.GetBondBetweenAtoms(*b).GetBondType().__str__() for b in bond_list]
            if bond_type_strs.count("DOUBLE") > max_double_in_6ring:
                for b in bond_list:
                    mw.RemoveBond(*b)
                    mw.AddBond(*b, Chem.rdchem.BondType.AROMATIC)

    # get new mol from modified mol
    conf = mw.GetConformer()
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(mw.GetNumAtoms())
    for i, atom in enumerate(mw.GetAtoms()):
        rd_atom = Chem.Atom(atom.GetAtomicNum())
        rd_mol.AddAtom(rd_atom)
        rd_coords = conf.GetAtomPosition(i)
        rd_conf.SetAtomPosition(i, rd_coords)
    rd_mol.AddConformer(rd_conf)

    for _i, bond in enumerate(mw.GetBonds()):
        bt = bond.GetBondType()
        node_i = bond.GetBeginAtomIdx()
        node_j = bond.GetEndAtomIdx()
        rd_mol.AddBond(node_i, node_j, bt)
    out_mol = rd_mol.GetMol()

    # check validity of the new mol
    mol_check = Chem.MolFromSmiles(Chem.MolToSmiles(out_mol))
    if mol_check:
        try:
            Chem.Kekulize(out_mol)
            del mol_copy
            return out_mol
        except Exception:
            del mol
            return mol_copy
    else:
        del mol
        return mol_copy


def save_sdf(mol_list: Sequence[Mol], save_name: str = "mol_gen.sdf") -> None:
    """
    Save molecules to an SDF file with simple computed properties.

    For each molecule, this computes:

    - Exact molecular weight (`MW`)
    - RDKit Crippen logP (`LOGP`)

    and writes them as SDF properties. Molecules are Kekulized before writing.

    Args:
        mol_list: Molecules to write.
        save_name: Output SDF path.
    """
    writer = Chem.SDWriter(save_name)
    writer.SetProps(["LOGP", "MW"])
    for i, mol in enumerate(mol_list):
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        mol.SetProp("MW", f"{mw:.2f}")
        mol.SetProp("LOGP", f"{logp:.2f}")
        mol.SetProp("_Name", f"No_{i}")
        Chem.Kekulize(mol)
        writer.write(mol)
    writer.close()


def check_alert_structure(mol: Mol, alert_smarts: str) -> bool:
    """
    Check whether a molecule matches a single alert SMARTS pattern.

    Args:
        mol: Molecule to search.
        alert_smarts: SMARTS pattern defining the alert.

    Returns:
        True if the molecule contains at least one match, otherwise False.
    """
    Chem.GetSSSR(mol)
    pattern = Chem.MolFromSmarts(alert_smarts)
    subs = mol.GetSubstructMatches(pattern)
    return len(subs) != 0


def check_alert_structures(mol: Mol, alert_smarts_list: Sequence[str]) -> bool:
    """
    Check whether a molecule matches any alert SMARTS patterns.

    Args:
        mol: Molecule to search.
        alert_smarts_list: A list/sequence of SMARTS patterns.

    Returns:
        True if any pattern matches, otherwise False.
    """
    Chem.GetSSSR(mol)
    patterns = [Chem.MolFromSmarts(sma) for sma in alert_smarts_list]
    for p in patterns:
        subs = mol.GetSubstructMatches(p)
        if len(subs) != 0:
            return True
    return False
