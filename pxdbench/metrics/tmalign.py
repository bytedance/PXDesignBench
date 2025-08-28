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
References:
- TM-align:
  Zhang, Y., & Skolnick, J. (2005). TM-align: A protein structure alignment algorithm based on TM-score.
  Nucleic Acids Research, 33(7), 2302–2309. [https://zhanggroup.org/TM-align/]
"""

import os
import re
import subprocess
from pathlib import Path

from Bio import PDB

from pxdbench.globals import TMALIGN_PATH


def extract_chain(input_pdb: str, chain_id: str, output_pdb: str):
    """
    Extract a specific chain from a PDB file.

    Args:
        input_pdb (str): Path to the input PDB file.
        chain_id (str): The chain ID to extract.
        output_pdb (str): Path to the output PDB file.
    """
    parser = PDB.PDBParser(QUIET=True)
    io = PDB.PDBIO()
    structure = parser.get_structure("structure", input_pdb)

    class ChainSelect(PDB.Select):
        def accept_chain(self, chain):
            return chain.id == chain_id

    io.set_structure(structure)
    io.save(output_pdb, select=ChainSelect())


def run_tmalign(file1: str, file2: str):
    """
    Run TM-align between two PDB files.

    Args:
        file1 (str): Path to the first PDB file.
        file2 (str): Path to the second PDB file.

    Returns:
        float or None: The TM-score between the two structures.
        Returns None if the TM-align command fails.
    """
    try:
        result = subprocess.run(
            [TMALIGN_PATH, file1, file2],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            if "TM-score=" in line and "Chain_1" in line:
                match = re.search(r"TM-score=\s*([0-9.]+)", line)
                if match:
                    return float(match.group(1))
    except subprocess.CalledProcessError as e:
        print(f"Error running TMalign on {file1} and {file2}: {e}")
    return None


def get_pdb_basename(pdb_path: str):
    assert pdb_path.endswith(".pdb")
    basename = os.path.basename(pdb_path)
    return basename[:-4]


def get_tm_score(
    pdb1: str, pdb2: str, chain1=None, chain2=None, keep_temp=False, temp_dir="tmp"
):
    """
    Calculate the TM-score between two PDB structures, optionally using specific chains.

    If specific chains are provided, extracts those chains into temporary files, runs TM-align,
    and optionally cleans up temporary files. If no chains are specified, runs TM-align directly
    on the input PDB files.

    Args:
        pdb1 (str): Path to the first PDB file.
        pdb2 (str): Path to the second PDB file.
        chain1 (str, optional): Chain ID to extract from the first PDB file. Defaults to None.
        chain2 (str, optional): Chain ID to extract from the second PDB file. Defaults to None.
        keep_temp (bool, optional): Whether to keep temporary chain files. Defaults to False.
        temp_dir (str, optional): Directory to store temporary chain files. Defaults to "tmp".

    Returns:
        float or None: The TM-score between the specified structures/chains.
        Returns None if TM-align execution fails or no valid TM-score is found.
    """
    if chain1 is None or chain2 is None:
        tm_score = run_tmalign(pdb1, pdb2)
    else:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        pdb1_chain = os.path.join(
            temp_dir, f"{get_pdb_basename(pdb1)}_chain{chain1}.pdb"
        )
        pdb2_chain = os.path.join(
            temp_dir, f"{get_pdb_basename(pdb2)}_chain{chain2}.pdb"
        )

        extract_chain(pdb1, chain1, pdb1_chain)
        extract_chain(pdb2, chain2, pdb2_chain)

        tm_score = run_tmalign(pdb1_chain, pdb2_chain)

        if not keep_temp:
            os.remove(pdb1_chain)
            os.remove(pdb2_chain)

    return tm_score
