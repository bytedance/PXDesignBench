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


def align_and_calculate_rmsd(file1, file2):
    parser = PDB.PDBParser(QUIET=True)
    structure1 = parser.get_structure("structure1", file1)
    structure2 = parser.get_structure("structure2", file2)

    coords1 = get_coordinates(structure1)
    coords2 = get_coordinates(structure2)

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


def Binder_align_and_calculate_rmsd(file1, file2, chain_id):
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
