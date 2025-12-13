# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script implements protein structure alignment (CA atoms) using the
Kabsch algorithm to compute optimal rotation and RMSD.

References:
- Kabsch W. (1976, 1978) A solution for the best rotation to relate two sets of vectors. Acta Crystallographica A.
"""

import numpy as np
from Bio import PDB


def get_coordinates(structure, chain_id=None):
    """
    Extract the coordinates of alpha carbon (CA) atoms from a protein structure.

    Args:
        structure (Bio.PDB.Structure.Structure): A protein structure object parsed by Bio.PDB.PDBParser.
        chain_id (str, optional): The ID of the specific protein chain to extract coordinates from.
                                  If None, coordinates are extracted from all chains. Defaults to None.

    Returns:
        numpy.ndarray: A 2D array where each row represents the 3D coordinates of a CA atom.
    """
    coords = []
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.id != chain_id:
                continue
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == "CA":
                        coords.append(atom.get_coord())
    return np.array(coords)


def kabsch_algorithm(P, Q):
    """
    Apply the Kabsch algorithm to find the optimal rotation matrix that aligns two sets of points.

    Args:
        P (numpy.ndarray): The first set of points with shape (N, 3)
        Q (numpy.ndarray): The second set of points with shape (N, 3)

    Returns:
        tuple: A tuple containing the rotation matrix (R), centroid of P (C_P), and centroid of Q (C_Q).
    """
    # Centroid of P and Q
    C_P = np.mean(P, axis=0)
    C_Q = np.mean(Q, axis=0)

    # Center the points
    P_centered = P - C_P
    Q_centered = Q - C_Q

    # Covariance matrix
    H = np.dot(P_centered.T, Q_centered)

    try:
        # Singular value decomposition
        U, S, Vt = np.linalg.svd(H)

        # Rotation matrix
        R = np.dot(Vt.T, U.T)

        # Special reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

    except np.linalg.LinAlgError:
        print("Warning: SVD did not converge. Returning identity rotation.")
        R = np.eye(3)  # Fallback to identity rotation

    return R, C_P, C_Q


def calculate_rmsd(P, Q):
    diff = P - Q
    return np.sqrt(np.sum(diff * diff) / len(P))


def _is_standard_residue(residue):
    hetflag = residue.id[0]  # ' '=standard, 'H_'=hetero/water/ligand
    return hetflag == " "


def _choose_altloc(atom_list):
    """Pick one altloc variant for a duplicated atom (e.g., CA).
    Preference order: highest occupancy; tie-breaker: altloc 'A' or blank.
    """
    if len(atom_list) == 1:
        return atom_list[0]
    best = max(
        atom_list,
        key=lambda a: (
            a.get_occupancy() or 0.0,
            1 if a.get_altloc() in ("A", " ") else 0,
        ),
    )
    return best


def _residue_key(chain, residue):
    """Build a stable residue key using (chain_id, resseq, icode)."""
    het, resseq, icode = residue.get_id()
    return (chain.id, int(resseq), (icode or "").strip())


def _collect_ca_coords(structure, chain_ids=None):
    """
    Collect CA coordinates keyed by (chain_id, resseq, icode).

    Returns
    -------
    dict[(chain_id, resseq, icode)] -> np.ndarray shape (3,), float64

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
    chain_ids : Iterable[str] | None Select multiple chains.
    """
    chain_id_set = set(chain_ids) if chain_ids is not None else None

    idx = {}
    for model in structure:
        for chain in model:
            if chain_id_set is not None:
                if chain.id not in chain_id_set:
                    continue

            for res in chain:
                if not _is_standard_residue(res):
                    continue
                ca_atoms = [a for a in res if a.get_name() == "CA"]
                if not ca_atoms:
                    continue
                ca = _choose_altloc(ca_atoms)
                key = _residue_key(chain, res)
                idx[key] = ca.get_coord().astype(np.float64)
    return idx


def align_and_calculate_rmsd(file1, file2):
    """
    Align two protein structures based on their CA atoms and calculate RMSD.

    Args:
        file1 (str): Path to the first PDB file.
        file2 (str): Path to the second PDB file.

    Returns:
        float or None: The RMSD value between the aligned structures.
        Returns None if the number of CA atoms in the two structures differs.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure1 = parser.get_structure("structure1", file1)
    structure2 = parser.get_structure("structure2", file2)

    coords1 = get_coordinates(structure1)
    coords2 = get_coordinates(structure2)

    if len(coords1) != len(coords2):
        print(
            "[WARNING] The lengths of coord1 and coord2 are different. There may exist missing atoms!"
        )
        orig_num_atoms = len(coords1), len(coords2)
        coords1 = _collect_ca_coords(structure1)
        coords2 = _collect_ca_coords(structure2)
        # Use only residues present in BOTH structures
        common_keys = sorted(set(coords1.keys()) & set(coords2.keys()))
        if len(common_keys) < 3:
            print(f"[WARNING] common CA pairs < 3 (got {len(common_keys)}). ")
            return None
        coords1 = np.vstack([coords1[k] for k in common_keys])
        coords2 = np.vstack([coords2[k] for k in common_keys])
        matched_num_atoms = len(coords1), len(coords2)
        print(
            f"Orig num atoms: {orig_num_atoms} Matched num atoms: {matched_num_atoms}"
        )

    R, C_P, C_Q = kabsch_algorithm(coords1, coords2)

    # Apply rotation and translation
    coords2_aligned = np.dot(coords2 - C_Q, R) + C_P

    rmsd = calculate_rmsd(coords1, coords2_aligned)
    return rmsd


def Binder_align_and_calculate_rmsd(file1, file2, chain_id):
    """
    Align two protein structures based on their CA atoms, with one structure's specific chain, and calculate RMSD.

    Args:
        file1 (str): Path to the first PDB file.
        file2 (str): Path to the second PDB file.
        chain_id (str): The ID of the specific protein chain to extract coordinates from.

    Returns:
        float or None: The RMSD value between the aligned structures.
        Returns None if the number of CA atoms in the two structures differs.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure1 = parser.get_structure("structure1", file1)
    structure2 = parser.get_structure("structure2", file2)

    coords1 = get_coordinates(structure1)
    coords2 = get_coordinates(structure2, chain_id)
    if len(coords1) != len(coords2):
        print(
            "[WARNING] The lengths of coord1 and coord2 are different. There may exist missing atoms!"
        )
        return None

    R, C_P, C_Q = kabsch_algorithm(coords1, coords2)

    # Apply rotation and translation
    coords2_aligned = np.dot(coords2 - C_Q, R) + C_P

    rmsd = calculate_rmsd(coords1, coords2_aligned)
    return rmsd


def _list_chain_ids(structure):
    """Return chain IDs in file order (first model only)."""
    model = next(structure.get_models())
    return [ch.id for ch in model]


def _coords_for_chain_ids(structure, chain_ids):
    """Stack CA coords for the given chain IDs (skip empty chains safely)."""
    chunks = []
    for cid in chain_ids:
        arr = get_coordinates(structure, chain_id=cid)
        if arr.size:
            chunks.append(arr)
    if not chunks:
        return np.empty((0, 3), dtype=float)
    return np.vstack(chunks)


def align_and_calculate_target_rmsd(file1, file2, n=None):
    parser = PDB.PDBParser(QUIET=True)
    structure1 = parser.get_structure("structure1", file1)
    structure2 = parser.get_structure("structure2", file2)

    chains1 = _list_chain_ids(structure1)
    chains2 = _list_chain_ids(structure2)

    if n is None:
        n = len(chains1)
    if n > len(chains2):
        print(f"[WARNING] file2 has only {len(chains2)} chains; capping n to that.")
        n = len(chains2)

    ids1 = chains1[:n]
    ids2 = chains2[:n]

    coords1 = _coords_for_chain_ids(structure1, ids1)
    coords2 = _coords_for_chain_ids(structure2, ids2)

    if len(coords1) != len(coords2):
        print(
            "[WARNING] The lengths of coord1 and coord2 are different. "
            "Trying residue-key matching fallback."
        )

        idx1 = _collect_ca_coords(structure1, chain_ids=ids1)
        idx2 = _collect_ca_coords(structure2, chain_ids=ids2)
        common_keys = sorted(set(idx1.keys()) & set(idx2.keys()))
        if len(common_keys) < 3:
            print(f"[WARNING] common CA pairs < 3 (got {len(common_keys)}).")
            return None
        coords1 = np.vstack([idx1[k] for k in common_keys])
        coords2 = np.vstack([idx2[k] for k in common_keys])
        print(f"Matched num CA atoms after fallback: {(len(coords1), len(coords2))}")

    R, C_P, C_Q = kabsch_algorithm(coords1, coords2)

    # Apply rotation and translation
    coords2_aligned = np.dot(coords2 - C_Q, R) + C_P

    rmsd = calculate_rmsd(coords1, coords2_aligned)
    return rmsd
