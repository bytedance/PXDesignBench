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

from eval_design.globals import AF2_PARAMS_PATH
from eval_design.metrics.Kalign import Binder_align_and_calculate_rmsd
from eval_design.tools.af2.af2_utils import add_cyclic_offset
from eval_design.utils import concat_dict_values, seed_everything

logger = logging.getLogger(__name__)


def predict_binder_monomer(
    prediction_model,
    sequence,
    design_name,
    model_indices,
    save_dir,
):
    sequence = re.sub(r"[^A-Z]", "", sequence.upper())
    prediction_model.set_seq(sequence)
    prediction_stats = {}

    for model_num in model_indices:
        output_name = f"{design_name}_model{model_num + 1}"
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
            prediction_model.predict(models=[model_num], num_recycles=3, verbose=True)
            metrics = copy_dict(prediction_model.aux["log"])
            stats = {
                "pLDDT_MONOMER": round(metrics["plddt"], 3),
                "pTM_MONOMER": round(metrics["ptm"], 3),
                "pAE_MONOMER": round(metrics["pae"], 3),
            }
            # save pdb and stats
            prediction_model.save_pdb(output_pdb)
            with open(output_stats_json, "w") as f:
                json.dump(stats, f)
        prediction_stats[model_num] = stats

    return prediction_stats


def binder_only_prediction(
    save_dir,
    design_pdb_dir,
    data_list,
    af2_cfg,
    binder_chain="B",
    verbose=True,
    is_cyclic=False,
):

    clear_mem()
    prediction_model = mk_afdesign_model(
        protocol="hallucination",
        use_templates=False,
        initial_guess=False,
        use_initial_atom_pos=False,
        num_recycles=3,
        data_dir=AF2_PARAMS_PATH,
        use_multimer=af2_cfg["use_multimer"],
    )

    os.makedirs(save_dir, exist_ok=True)

    results = []
    length_prev = -1
    for item in data_list:
        name = item["name"]
        seq = item["sequence"]
        seq_idx = item["seq_idx"]
        binder_len = len(seq)

        # Only compile when the inference length changes
        if length_prev != binder_len:
            prediction_model.prep_inputs(
                length=binder_len,
            )
            length_prev = binder_len
            if is_cyclic:
                add_cyclic_offset(prediction_model)
        design_name = f"{name}_seq{seq_idx}_MONOMER_ONLY"
        design_complex_name = f"{name}_seq{seq_idx}"
        stats = predict_binder_monomer(
            prediction_model, seq, design_name, af2_cfg["model_ids"], save_dir
        )

        stat_list = []
        for model_id in af2_cfg["model_ids"]:
            s = stats[model_id]
            pred_binder_pdb = os.path.join(
                save_dir, f"{design_name}_model{model_id + 1}.pdb"
            )
            pred_complex_pdb = os.path.join(
                save_dir, f"{design_complex_name}_model{model_id + 1}.pdb"
            )
            if os.path.isfile(pred_complex_pdb):
                # af2 binder chain is "B"
                bound_unbound_RMSD = round(
                    Binder_align_and_calculate_rmsd(
                        pred_binder_pdb, pred_complex_pdb, "B"
                    ),
                    2,
                )
            else:
                bound_unbound_RMSD = None

            # compute predict-design RMSD
            ori_design_pdb = os.path.join(design_pdb_dir, name + ".pdb")
            if os.path.isfile(ori_design_pdb):
                binder_rmsd = round(
                    Binder_align_and_calculate_rmsd(
                        pred_binder_pdb, ori_design_pdb, binder_chain
                    ),
                    2,
                )
            else:
                binder_rmsd = None

            s["bound_unbound_RMSD"] = bound_unbound_RMSD
            s["af2_binder_pred_design_rmsd"] = binder_rmsd
            stat_list.append(s)
        stat = concat_dict_values(stat_list)
        if verbose:
            print(f"{name}-seq{seq_idx}, {stat}")
        results.append(stat)
    return results


def main():
    parser = argparse.ArgumentParser(description="AF2 Binder Monomer Prediction")
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
        results = binder_only_prediction(
            save_dir=input_data["save_dir"],
            design_pdb_dir=input_data["design_pdb_dir"],
            data_list=input_data["data_list"],
            binder_chain=input_data["binder_chain"],
            af2_cfg=input_data["af2_cfg"],
            verbose=True,
            is_cyclic=input_data["is_cyclic"],
        )

        with open(args.output, "w") as f:
            json.dump(results, f)

        print(f"Successfully completed AF2 binder only prediction!")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
