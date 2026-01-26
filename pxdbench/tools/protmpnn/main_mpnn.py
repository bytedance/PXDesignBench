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

import numpy as np
from colabdesign.mpnn import clear_mem, mk_mpnn_model
from ml_collections import ConfigDict

from pxdbench.tools.biopython_utils import get_interface_residue_id, hotspot_residues
from pxdbench.utils import extract_chain_sequence, seed_everything

AA_ALPHABET = "ARNDCQEGHILKMFPSTWYV"


def _load_bias_aa_map(bias_aa_json):
    if not bias_aa_json:
        return None
    with open(bias_aa_json, "r") as f:
        bias_map = json.load(f)
    if not isinstance(bias_map, dict):
        raise ValueError("bias_aa_json must contain a JSON object mapping AAs to values.")
    return bias_map


def _resolve_bias_alphabet(num_cols):
    if num_cols == len(AA_ALPHABET):
        return AA_ALPHABET
    if num_cols == len(AA_ALPHABET) + 1:
        return AA_ALPHABET + "X"
    raise ValueError(
        f"Unsupported bias width {num_cols}; expected {len(AA_ALPHABET)} or "
        f"{len(AA_ALPHABET) + 1}."
    )


def _normalize_bias_matrix(bias_matrix, num_cols):
    if isinstance(bias_matrix, list):
        bias_matrix = np.array(bias_matrix, dtype=np.float32)
    if not isinstance(bias_matrix, np.ndarray):
        raise ValueError("bias_matrix_json must contain a JSON array.")
    if bias_matrix.ndim != 2 or bias_matrix.shape[1] != num_cols:
        raise ValueError(
            f"bias_matrix_json must be a (length, {num_cols}) matrix."
        )
    return bias_matrix


def _load_bias_matrix(bias_matrix_json, num_cols):
    if not bias_matrix_json:
        return None
    with open(bias_matrix_json, "r") as f:
        bias_matrix = json.load(f)
    return _normalize_bias_matrix(bias_matrix, num_cols)


def _apply_bias_inputs(mpnn_model, bias_aa_json, bias_matrix_json):
    bias_map = _load_bias_aa_map(bias_aa_json)
    bias_store = mpnn_model._inputs["bias"]
    num_cols = bias_store.shape[1]
    alphabet = _resolve_bias_alphabet(num_cols)
    bias_matrix = _load_bias_matrix(bias_matrix_json, num_cols)
    if bias_map is not None:
        bias = np.zeros((bias_store.shape[0], num_cols), dtype=np.float32)
        aa_order = {aa: idx for idx, aa in enumerate(alphabet)}
        for aa, value in bias_map.items():
            if aa not in aa_order:
                raise ValueError(
                    f"Unknown amino acid '{aa}' in bias_aa_json. "
                    f"Supported letters: {alphabet}."
                )
            bias[:, aa_order[aa]] += float(value)
        mpnn_model._inputs["bias"] += bias
    if bias_matrix is not None:
        if bias_matrix.shape[0] != mpnn_model._inputs["S"].shape[1]:
            raise ValueError(
                "bias_matrix_json length must match the number of residues."
            )
        mpnn_model._inputs["bias"] += bias_matrix


def get_pdb_basename(pdb_path: str):
    assert pdb_path.endswith(".pdb")
    basename = os.path.basename(pdb_path)
    return basename[:-4]


def design_monomer(
    pdb_dir: str,
    pdb_names: list[str],
    num_samples: int,
    mpnn_cfg: ConfigDict,
    if_print=True,
):
    """
    Design sequences for monomer proteins using ProteinMPNN.

    Args:
        pdb_dir (str): Directory containing input PDB files.
        pdb_names (list[str]): List of PDB base names (without '.pdb') to process.
        num_samples (int): Number of sequences to generate per PDB.
        mpnn_cfg (ConfigDict): ProteinMPNN configuration (temperature, weights, etc.).
        if_print (bool, optional): Whether to print progress. Defaults to True.

    Returns:
        list[dict]: List of design results with keys 'name' (PDB name), 'seq_idx' (sequence index),
                    and 'sequence' (designed amino acid sequence).
    """
    clear_mem()
    mpnn_model = mk_mpnn_model(
        backbone_noise=0.0,
        model_name="v_48_020",
        weights=mpnn_cfg.weights,
    )

    final_result = []

    for name in pdb_names:
        pdb_path = os.path.join(pdb_dir, name + ".pdb")
        # Prepare MPNN input
        mpnn_model.prep_inputs(
            pdb_filename=pdb_path,
            chain="A",
        )
        _apply_bias_inputs(
            mpnn_model,
            mpnn_cfg.get("bias_aa_json", ""),
            mpnn_cfg.get("bias_matrix_json", ""),
        )

        if if_print:
            print(f"{pdb_path} is done")

        temperature = mpnn_cfg.temperature
        if isinstance(temperature, str):
            if temperature == "auto":
                temperature = 0.0001 if num_samples > 1 else 0.1
                print(
                    f"Use temperature {temperature} for num_samples being {num_samples}"
                )
            else:
                temperature = float(temperature)
        else:
            raise ValueError(f"Unknown temperature {temperature}")

        # Run MPNN sampling
        mpnn_sequences = mpnn_model.sample(
            temperature=temperature,
            num=num_samples,
            batch=1,
        )

        # Collect sequences
        for i, seq in enumerate(mpnn_sequences["seq"]):
            final_result.append(
                {"name": name, "seq_idx": i, "sequence": seq.split("/")[-1]}
            )

    if if_print:
        print("finished all Sequence Design")
    return final_result


def design_binder(
    pdb_dir: str,
    pdb_names: list[str],
    num_samples: int,
    binder_chains: list[str],
    cond_chains: list[str],
    mpnn_cfg: ConfigDict,
    if_print=True,
):
    """
    Design sequences for binder proteins using ProteinMPNN.

    Args:
        pdb_dir (str): Directory containing input PDB files.
        pdb_names (list[str]): List of PDB base names (without '.pdb') to process.
        num_samples (int): Number of sequences to generate per PDB.
        binder_chains (list[str]): List of binder chain IDs.
        cond_chains (list[str]): List of conditional chain IDs.
        mpnn_cfg (ConfigDict): ProteinMPNN configuration (temperature, weights, etc.).
        if_print (bool, optional): Whether to print progress. Defaults to True.

    Returns:
        list[dict]: List of design results with keys 'name' (PDB name), 'seq_idx' (sequence index),
                    and 'sequence' (designed amino acid sequence).
    """

    clear_mem()
    mpnn_model = mk_mpnn_model(
        backbone_noise=0.0,
        model_name="v_48_020",
        weights=mpnn_cfg.weights,
    )

    final_result = []

    for name in pdb_names:
        pdb_path = os.path.join(pdb_dir, name + ".pdb")
        # Prepare MPNN input
        if len(binder_chains) > 1:
            raise ValueError(f"Only support one-chain binders, but got {binder_chains}")
        if mpnn_cfg.fix_interface:
            interacting_residues = hotspot_residues(
                pdb_path=pdb_path, binder_chain=binder_chains[0]
            )  # Hardcode, only take the first binder chains
            if len(interacting_residues) > 0:
                fix_pos = get_interface_residue_id(
                    interacting_residues=interacting_residues,
                    binder_chain=binder_chains[0],
                )
            else:
                fix_pos = ",".join(cond_chains)

        else:
            fix_pos = ",".join(cond_chains)
        mpnn_model.prep_inputs(
            pdb_filename=pdb_path,
            chain=",".join(cond_chains + binder_chains),
            fix_pos=fix_pos,
            rm_aa=mpnn_cfg.rm_aa,
        )
        _apply_bias_inputs(
            mpnn_model,
            mpnn_cfg.get("bias_aa_json", ""),
            mpnn_cfg.get("bias_matrix_json", ""),
        )

        if if_print:
            print(f"{pdb_path} is done")

        temperature = mpnn_cfg.temperature
        if isinstance(temperature, str):
            if temperature == "auto":
                temperature = 0.0001 if num_samples > 1 else 0.1
                print(
                    f"Use temperature {temperature} for num_samples being {num_samples}"
                )
            else:
                temperature = float(temperature)
        else:
            raise ValueError(f"Unknown temperature {temperature}")

        # Run MPNN sampling
        mpnn_sequences = mpnn_model.sample(
            temperature=temperature,
            num=num_samples,
            batch=1,
        )

        # Collect sequences
        for i, seq in enumerate(mpnn_sequences["seq"]):
            final_result.append(
                {"name": name, "seq_idx": i, "sequence": seq.split("/")[-1]}
            )

    if if_print:
        print("finished all Sequence Design")

    return final_result


def get_gt_sequence(pdb_dir: str, pdb_names: list[str], binder_chain="B"):
    """
    Get ground truth sequences for binder proteins from PDB files.

    Args:
        pdb_dir (str): Directory containing input PDB files.
        pdb_names (list[str]): List of PDB base names (without '.pdb') to process.
        binder_chain (str, optional): Chain ID of the binder protein. Defaults to "B".

    Returns:
        list[dict]: List of design results with keys 'name' (PDB name), 'seq_idx' (sequence index),
                    and 'sequence' (ground truth amino acid sequence).
    """

    final_result = []
    for name in pdb_names:
        sequence = []
        result = {}
        pdb_path = os.path.join(pdb_dir, name + ".pdb")
        seq = extract_chain_sequence(pdb_path, chain_id=binder_chain)
        sequence.append(seq)
        result["name"] = name
        result["sequences"] = sequence
        final_result.append({"name": name, "seq_idx": 0, "sequence": sequence[0]})

    return final_result


def main():
    parser = argparse.ArgumentParser(description="ProteinMPNN Sequence Design")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    with open(args.input, "r") as f:
        input_data = json.load(f)

    if args.seed is not None:
        seed_everything(args.seed, deterministic=False)
    design_type = input_data["design_type"]

    try:
        if design_type == "monomer":
            result = design_monomer(
                pdb_dir=input_data["pdb_dir"],
                pdb_names=input_data["pdb_names"],
                num_samples=input_data["num_samples"],
                mpnn_cfg=ConfigDict(input_data["mpnn_cfg"]),
                if_print=True,
            )
        elif design_type == "binder":
            result = design_binder(
                pdb_dir=input_data["pdb_dir"],
                pdb_names=input_data["pdb_names"],
                num_samples=input_data["num_samples"],
                binder_chains=input_data["binder_chains"],
                cond_chains=input_data["cond_chains"],
                mpnn_cfg=ConfigDict(input_data["mpnn_cfg"]),
                if_print=True,
            )
        elif design_type == "gt":
            result = get_gt_sequence(
                pdb_dir=input_data["pdb_dir"],
                pdb_names=input_data["pdb_names"],
                binder_chain=input_data.get("binder_chain", "B"),
            )
        else:
            raise ValueError(f"Unknown design type: {design_type}")

        with open(args.output, "w") as f:
            json.dump(result, f)

        print(f"Successfully completed {design_type} design")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
