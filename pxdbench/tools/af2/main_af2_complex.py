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
import logging
import os
import re

from colabdesign import clear_mem, mk_afdesign_model
from colabdesign.shared.utils import copy_dict

from pxdbench.globals import AF2_PARAMS_PATH
from pxdbench.tools.af2.af2_utils import add_cyclic_offset
from pxdbench.utils import concat_dict_values, seed_everything

logger = logging.getLogger(__name__)


def predict_binder_structure(
    prediction_model,
    sequence: str,
    design_name: str,
    model_indices: list[int],
    save_dir: str,
):
    """
    Predict binder structure using AlphaFold2 and compute structural metrics.

    Args:
        prediction_model: Initialized ColabDesign AFDesign model instance.
        sequence (str): Amino acid sequence of the binder to predict.
        design_name (str): Unique identifier for the design (e.g., "pdbname_seq0").
        model_indices (list[int]): List of AlphaFold2 model indices to use (0-4).
        save_dir (str): Directory to save predicted PDB files and metrics.

    Returns:
        dict: Prediction statistics (pLDDT, pTM, i_pTM, etc.) for each model index.
    """
    sequence = re.sub(r"[^A-Z]", "", sequence.upper())
    prediction_stats = {}

    for model_num in model_indices:
        output_name = f"{design_name}_model{model_num+1}"
        output_pdb = os.path.join(save_dir, f"{output_name}.pdb")
        output_stats_json = os.path.join(save_dir, f"{output_name}.json")

        if os.path.exists(output_pdb) and os.path.exists(output_stats_json):
            print(
                f"Found existing {output_pdb} and {output_stats_json}. Will load from them."
            )
            # load stats
            with open(output_stats_json, "r") as f:
                stats = json.load(f)
            print(f"Loaded {output_stats_json}.")

        else:
            prediction_model.predict(
                seq=sequence, models=[model_num], num_recycles=3, verbose=True
            )
            metrics = copy_dict(prediction_model.aux["log"])
            stats = {
                "pLDDT": round(metrics["plddt"], 2),
                "pTM": round(metrics["ptm"], 2),
                "i_pTM": round(metrics["i_ptm"], 2),
                "pAE": round(metrics["pae"], 2),
                "i_pAE": round(metrics["i_pae"], 2),  # i_pae divdied by 31
                "unscaled_i_pAE": round(metrics["i_pae"] * 31, 2),  # raw i_pae
            }
            # save pdb and stats
            prediction_model.save_pdb(output_pdb)
            with open(output_stats_json, "w") as f:
                json.dump(stats, f)

        prediction_stats[model_num] = stats

    return prediction_stats


def complex_prediction(
    input_dir: str,
    save_dir: str,
    data_list: list[dict],
    cond_chain: str,
    binder_chain: str,
    af2_cfg,
    verbose=True,
    is_cyclic=False,
):
    """
    Run batch prediction for binder complexes using AlphaFold2.

    Args:
        input_dir (str): Directory containing input PDB files for target structures.
        save_dir (str): Directory to save prediction outputs (PDBs, metrics).
        data_list (list[dict]): List of design data with keys "name", "sequence", "seq_idx".
        cond_chain (str): Chain ID(s) of the target (conditioning) structure(s).
        binder_chain (str): Chain ID of the binder to design/predict.
        af2_cfg (dict): AlphaFold2 configuration (model indices, multimer usage, etc.).
        verbose (bool, optional): Whether to print progress. Defaults to True.
        is_cyclic (bool, optional): Whether the binder is cyclic (adds cyclic offset). Defaults to False.

    Returns:
        list[dict]: Aggregated prediction statistics for each design in data_list.
    """
    use_binder_template = af2_cfg["use_binder_template"]
    logger.info(f"Input use_binder_template: {use_binder_template}")

    clear_mem()
    prediction_model = mk_afdesign_model(
        protocol="binder",
        num_recycles=3,
        data_dir=AF2_PARAMS_PATH,
        use_multimer=af2_cfg["use_multimer"],
        use_initial_guess=af2_cfg["use_initial_guess"],
        use_initial_atom_pos=af2_cfg["use_initial_atom_pos"],
    )

    os.makedirs(save_dir, exist_ok=True)

    results = []
    for item in data_list:
        name = item["name"]
        seq = item["sequence"]
        seq_idx = item["seq_idx"]
        pdb_file = os.path.join(input_dir, f"{name}.pdb")
        if not os.path.exists(pdb_file):
            print(f"ERROR: {pdb_file} not found")
            continue

        prediction_model.prep_inputs(
            pdb_filename=pdb_file,
            chain=cond_chain,
            binder_chain=binder_chain,
            use_binder_template=use_binder_template,
            rm_target_seq=True,
            rm_target_sc=False,
            rm_template_ic=True,
        )
        if is_cyclic:
            add_cyclic_offset(prediction_model)

        design_name = f"{name}_seq{seq_idx}"
        stats = predict_binder_structure(
            prediction_model, seq, design_name, af2_cfg["model_ids"], save_dir
        )
        stat_list = []
        for model_id in af2_cfg["model_ids"]:
            s = stats[model_id]
            stat_list.append(s)
        stat = concat_dict_values(stat_list)
        if verbose:
            print(f"{name}-seq{seq_idx}, {stat}")
        results.append(stat)
    return results


def main():
    parser = argparse.ArgumentParser(description="AF2 Binder Complex Prediction")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    with open(args.input, "r") as f:
        input_data = json.load(f)

    # args = parser.parse_args()
    # model_ids = [int(x) for x in args.model_ids.split(",")]

    if args.seed is not None:
        seed_everything(args.seed, deterministic=False)

    try:
        results = complex_prediction(
            input_dir=input_data["input_dir"],
            save_dir=input_data["save_dir"],
            data_list=input_data["data_list"],
            cond_chain=input_data["cond_chain"],
            binder_chain=input_data["binder_chain"],
            af2_cfg=input_data["af2_cfg"],
            verbose=True,
            is_cyclic=input_data["is_cyclic"],
        )

        with open(args.output, "w") as f:
            json.dump(results, f)

        print(f"Successfully completed AF2 binder complex prediction!")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
