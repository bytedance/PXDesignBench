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

import biotite.structure.io.pdbx as pdbx
import numpy as np
from Bio.PDB import PDBParser
from biotite.structure import annotate_sse
from biotite.structure.io.pdb import PDBFile


def cacl_secondary_structure(path, binder_chain=None):
    if path.endswith(".cif"):
        pdb_name = pdbx.PDBxFile.read(path)
        atom_array = pdbx.get_structure(pdb_name, model=1)
    elif path.endswith(".pdb"):
        pdb_name = PDBFile.read(path)
        atom_array = pdb_name.get_structure()[0]
    else:
        raise ValueError

    if binder_chain is not None:
        chain_mask = atom_array.chain_id == binder_chain
        atom_array = atom_array[chain_mask]
    sse_array = annotate_sse(atom_array)

    count_a = np.sum(sse_array == "a")
    count_b = np.sum(sse_array == "b")
    count_c = np.sum(sse_array == "c")
    total = len(sse_array)
    return (
        round(count_a / total, 3),
        round(count_b / total, 3),
        round(count_c / total, 3),
    )


def cacl_ref_rg(num_res):
    return 2.38 * num_res**0.365


def get_chain_rg(pdb_file, chain_id, atom_name="CA"):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_file)
    coords = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for res in chain:
                    if atom_name in res:
                        atom = res[atom_name]
                        coords.append(atom.coord)
    coords = np.array(coords)
    if coords.size == 0:
        raise ValueError("No atoms found!")
    center = coords.mean(axis=0)
    num_res = len(coords)
    rg = np.sqrt(((coords - center) ** 2).sum(axis=1).mean())
    return round(rg, 2), round(rg / cacl_ref_rg(num_res), 4)
