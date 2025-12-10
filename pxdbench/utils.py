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

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from glob import glob
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from Bio.PDB import MMCIFParser as BioMMCIFParser
from Bio.PDB import *
from Bio.PDB.Polypeptide import is_aa
from biotite.structure import get_residue_starts
from natsort import natsorted

from protenix.data import ccd
from protenix.data.parser import MMCIFParser

three_to_one = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic=True applies to CUDA convolution operations, and nothing else.
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True) affects all the normally-nondeterministic operations listed here https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html?highlight=use_deterministic#torch.use_deterministic_algorithms
        torch.use_deterministic_algorithms(True)
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Error")


def convert_cif_to_pdb(
    cif_path: str,
    out_pdb_path: str = None,
    binder_chains: list[str] = None,
    trim_chain_ids=True,
    resname_mapping: dict = {"xpb": "GLY"},  # can replace "xpb" with "GLY"
) -> None:
    """
    Convert a CIF file to a PDB file.
    Args:
        cif_path: Path to the CIF file.
        out_pdb_path: Path to save the PDB file. If None, will save to the same directory as the CIF file.
        binder_chains: List of chain IDs to consider as binders. If None, will use all chains.
        trim_chain_ids: If True, will trim the chain IDs to the first character.
        resname_mapping: Dictionary of residue names to replace. If None, will not replace residue names.
    Returns:
        List of condition chain IDs and list of binder chain IDs.
    """
    if out_pdb_path is None:
        assert cif_path.endswith(".cif")
        out_pdb_path = cif_path[: -len(".cif")] + ".pdb"

    parser = BioMMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_path)

    if trim_chain_ids:
        # Collect and check for chain ID conflicts
        original_ids = []
        trimmed_ids = []
        for model in structure:
            for chain in model:
                original_ids.append(chain.id)
                trimmed_ids.append(chain.id[0])
        if len(set(trimmed_ids)) < len(set(original_ids)):
            raise ValueError(
                "Chain ID collision detected after trimming to 1 character:\n"
                f"Original IDs: {sorted(set(original_ids))}\n"
                f"Trimmed IDs: {sorted(set(trimmed_ids))}"
            )

    new_cond_chains, new_binder_chains = [], []
    for model in structure:
        for chain in model:
            new_chain_id = chain.id[0] if trim_chain_ids else chain.id
            if binder_chains is not None and chain.id in binder_chains:
                new_binder_chains.append(new_chain_id)
            else:
                new_cond_chains.append(new_chain_id)
            if trim_chain_ids:
                chain.id = chain.id[0]  # Trim to first character
            if resname_mapping is not None:
                for res in chain:
                    if res.resname in resname_mapping:
                        res.resname = resname_mapping[res.resname]
    if binder_chains is not None and len(new_binder_chains) == 0:
        raise ValueError(f"binder chains {binder_chains} not found in the cif file.")

    if os.path.exists(out_pdb_path):
        print(
            f"[WARNING] PDB file {out_pdb_path} already exists when trying to convert a CIF file to it"
        )
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb_path)
    return new_cond_chains, new_binder_chains


def find_cif_files(folder_path):
    pdb_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".cif"):
            full_path = os.path.join(folder_path, filename)
            pdb_files.append(full_path)
    return pdb_files


def find_cond_chains(cif_path):
    # Note: we only consider one binder chain, which is also the last chain!
    mmcif_parser = BioMMCIFParser()
    structure = mmcif_parser.get_structure("protein", cif_path)
    cond_chains = [chain.id for chain in structure[0]]
    return cond_chains[:-1]


def find_binder_chains(cif_path, condition_chains):
    try:
        mmcif_parser = BioMMCIFParser()
        structure = mmcif_parser.get_structure("protein", cif_path)
    except:
        print("find binder chains fail: ", cif_path, condition_chains)
        raise ValueError()

    # only consider polypeptide as binder, we do not need it now actually because there is a "filter" field in the json file
    # TODO: need a more general way to determine polypeptide
    all_chain_ids = []
    for chain in structure[0]:
        residues = [
            res for res in chain if res.resname == "xpb" or is_aa(res, standard=False)
        ]
        if len(residues) > 0:
            all_chain_ids.append(chain.id)
    assert all(c in all_chain_ids for c in condition_chains)
    binder_chains = list(set(all_chain_ids) - set(condition_chains))
    return binder_chains


def convert_cifs_to_pdbs(
    input_dir: str,
    out_pdb_dir: str = None,
    condition_chains: list[str] = None,
    resname_mapping: dict = {"xpb": "GLY"},
):
    """
    Converts all mmCIF (.cif) files in a directory to PDB format.

    This function scans the input directory for `.cif` files, infers binder chains
    from the first file, and converts all files to `.pdb` format using the same
    binder chains. The output PDB files are saved in the specified output directory.

    Args:
        input_dir (str): Path to the directory containing .cif files.
        out_pdb_dir (Optional[str]): Directory to save the converted .pdb files.
            If None, a `converted_pdbs` subdirectory will be created in `input_dir`.
        condition_chains (list[str]): List of condition chain IDs used to infer binder chains.
        resname_mapping (dict): Dictionary of residue names to replace.

    Returns:
        tuple[list[str], str, list[str], list[str]]:
            - Output PDB dir
            - List of output PDB names.
            - List of condition chain IDs.
            - List of binder chain IDs.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(input_dir)
    assert os.path.isdir(input_dir), "The input should be a directory"
    all_cif_files = find_cif_files(input_dir)
    if len(all_cif_files) == 0:
        print(f"[WARNING] Can not find cif files in {input_dir}")
        return [], None, None
    if condition_chains is None:
        condition_chains = find_cond_chains(all_cif_files[0])
    binder_chains = find_binder_chains(all_cif_files[0], condition_chains)
    if out_pdb_dir is None:
        out_pdb_dir = os.path.join(input_dir, "converted_pdbs")
    os.makedirs(out_pdb_dir, exist_ok=True)

    pdb_names = []
    for cif_file in all_cif_files:
        assert cif_file.endswith(".cif")
        cur_binder_chains = find_binder_chains(cif_file, condition_chains)
        assert set(binder_chains) == set(cur_binder_chains), (
            f"Binder chains in {cif_file} differ from those in the first file: "
            f"{set(cur_binder_chains)} != {set(binder_chains)}"
        )
        prefix = os.path.basename(cif_file)[: -len(".cif")]
        pdb_names.append(prefix)
        pdb_file = os.path.join(out_pdb_dir, f"{prefix}.pdb")
        new_cond_chains, new_binder_chains = convert_cif_to_pdb(
            cif_path=cif_file,
            out_pdb_path=pdb_file,
            binder_chains=binder_chains,
            resname_mapping=resname_mapping,
        )
    pdb_names = sorted(pdb_names)
    return out_pdb_dir, pdb_names, new_cond_chains, new_binder_chains


def merge_list_of_dicts_on_key(
    list1: List[Dict[str, Any]], list2: List[Dict[str, Any]], key: str
) -> List[Dict[str, Any]]:
    """
    Merge two lists of dictionaries based on a common key.

    Args:
        list1: First list of dictionaries.
        list2: Second list of dictionaries.
        key: The key to merge on.

    Returns:
        A list of merged dictionaries.
    """
    index2 = {d[key]: d for d in list2}
    merged = []
    for d1 in list1:
        k = d1[key]
        if k in index2:
            merged.append({**d1, **index2[k]})
        else:
            merged.append(d1)
    return merged


def concat_dict_values(dict_list: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    result = defaultdict(list)
    for d in dict_list:
        for k, v in d.items():
            result[k].append(v)
    return dict(result)


def save_eval_results(
    sample_df,
    summary_dict,
    root_dir,
    sample_fn: str = "sample_level_output.csv",
    summary_fn: str = "summary_output.json",
):
    sample_save_path = os.path.join(root_dir, sample_fn)
    summary_save_path = os.path.join(root_dir, summary_fn)
    sample_df.to_csv(sample_save_path, index=False)
    with open(summary_save_path, "w") as f:
        json.dump(summary_dict, f, indent=4)
    return sample_save_path, summary_save_path


def extract_chain_sequence(pdb_file, chain_id="R"):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_file)
    for model in structure:
        if chain_id in model:
            chain = model[chain_id]
            seq = []
            current_resid = None
            for residue in chain:
                # Fill any numbering gaps with 'X'
                if current_resid is not None:
                    gap = residue.id[1] - current_resid - 1
                    seq.extend("X" * gap)
                current_resid = residue.id[1]

                try:
                    seq.append(three_to_one[residue.get_resname()])
                except KeyError:
                    seq.append("X")  # non-standard residue

            return "".join(seq)


def extract_chain_sequence_from_mmcif(cif_file, chain_id="B"):
    parser = MMCIFParser(cif_file)
    atom_array = parser.get_structure(
        altloc="first", model=1, bond_lenth_threshold=None
    )
    chain_atom_array = atom_array[atom_array.chain_id == chain_id]
    starts = get_residue_starts(chain_atom_array, add_exclusive_stop=True)
    res_names = chain_atom_array.res_name[starts[:-1]].tolist()
    seq = ccd.res_names_to_sequence(res_names)
    return seq


def prepare_tasks(task_json_path, input_dir, save_dir=None, task_indices=None):
    with open(task_json_path, "r") as f:
        data = json.load(f)
    if task_indices is not None:
        data = [data[i] for i in task_indices]

    inputs = []
    for x in data:
        if "condition" in x or "sequences" in x:
            task = "binder"
        else:
            task = "monomer"

        # only consider one seed for now
        input_data_dir = glob(
            os.path.join(input_dir, x["name"], "seed_*", "predictions")
        )
        if len(input_data_dir) == 0:
            print(f"Could not find data to eval for name: {x['name']}")
            continue
        input_data_dir = input_data_dir[0]
        rel_path = os.path.relpath(input_data_dir, input_dir)
        out_dir = (
            input_data_dir if save_dir is None else os.path.join(save_dir, rel_path)
        )
        summary_path = os.path.join(out_dir, "summary_output.json")
        if os.path.exists(summary_path):
            print(f"Found existing {summary_path}, skip!")
            continue

        # cond_chains = x["condition"]["filter"]["chain_id"] if "condition" in x else []
        pdb_dir, pdb_names, cond_chains, binder_chains = convert_cifs_to_pdbs(
            input_data_dir,
            out_pdb_dir=os.path.join(out_dir, "converted_pdbs"),
        )

        if len(binder_chains) != 1:
            raise ValueError(
                f"Multiple binder chains are not supported for now! cond chains: {cond_chains}, binder chains: {binder_chains}"
            )

        inputs.append(
            {
                "task": task,
                "name": x["name"],
                "pdb_dir": pdb_dir,
                "pdb_names": pdb_names,
                "cond_chains": cond_chains,
                "binder_chains": binder_chains,
                "out_dir": out_dir,
            }
        )

    inputs = natsorted(inputs, key=lambda x: x["name"])
    for x in inputs:
        print(x)
    return inputs
