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

import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pxdbench.metrics import consistency
from pxdbench.tasks.base import BaseTask
from pxdbench.tools import esmfold
from pxdbench.tools.protmpnn.vanilla_mpnn_predictor import VanillaMPNNPredictor
from pxdbench.utils import save_eval_results

from .registry import register_task


@register_task("monomer")
class MonomerTask(BaseTask):
    def __init__(self, input_data, cfg, device_id: int, seed: int):
        """
        Initialize a MonomerTask instance.

        Args:
            input_data (dict): Task input parameters with PDB directory and names.
            cfg (dict): Configuration dictionary with task settings.
            device_id (int): GPU device ID (-1 for CPU).
            seed (int): Random seed for reproducibility.
        """
        self.task_type = "monomer"
        self.task_name = input_data.get("name", "monomer")
        self.eval_diversity = cfg.get("eval_diversity", False)
        super().__init__(input_data, cfg, device_id, seed)

    def get_target_fn(self, item):
        return item["name"] + f"_seq{item['seq_idx']}.pdb"

    def prepare_consistency_inputs(self, results, folding_dir):
        inputs = {}
        for item in results:
            name = f"{item['name']}_seq{item['seq_idx']}"
            inputs[name] = {
                "source_file": os.path.join(self.pdb_dir, item["name"] + ".pdb"),
                "target_file": os.path.join(folding_dir, self.get_target_fn(item)),
            }
        return inputs

    def design_sequence(self, verbose=True):
        """
        Design monomer sequences using Vanilla MPNN.

        Initializes a VanillaMPNNPredictor and uses it to generate sequences for monomer proteins.

        Args:
            verbose (bool, optional): Whether to print detailed progress. Defaults to True.

        Returns:
            list[dict]: List of design results with "name", "seq_idx", and "sequence" keys.
        """
        mpnn_predictor = VanillaMPNNPredictor(
            self.cfg.tools.mpnn,
            device_id=self.device_id,
            verbose=verbose,
            seed=self.seed,
        )
        results = mpnn_predictor.design_monomer(
            self.pdb_dir, self.pdb_names, self.num_seqs
        )
        return results

    def run(self):
        """
        Execute the complete monomer design evaluation workflow.

        Workflow steps:
        1. Design sequences via design_sequence()
        2. Predict structures using ESMFold and evaluate self consistency
        3. Calculate secondary structure metrics
        4. Compute diversity and success rates based on scRMSD thresholds
        5. Save sample-level results to CSV and summary metrics to JSON

        Returns:
            dict: Task metadata and output file paths.
        """
        results = self.design_sequence()
        esmfold_model = esmfold.ESMFold(self.get_device())
        print("Load esmfold done!")
        folding_dir = os.path.join(self.out_dir, "esmfold")
        os.makedirs(folding_dir, exist_ok=True)

        for item in tqdm(results, desc="ESMFold eval"):
            pdb_str, plddt = esmfold_model.predict([item["sequence"]])
            assert len(pdb_str) == 1 and len(plddt) == 1
            with open(os.path.join(folding_dir, self.get_target_fn(item)), "w") as f:
                f.write(pdb_str[0])
            item["plddt"] = plddt[0]

        inputs = self.prepare_consistency_inputs(results, folding_dir)
        outputs = consistency.self_consistency(inputs)
        for item in results:
            consistency_key = f"{item['name']}_seq{item['seq_idx']}"
            item.update(outputs[consistency_key])

        self.cal_secondary(results, chain_id="A")

        overall = {}
        for threshold in [2, 5]:
            success_names = []
            for item in results:
                if item["scRMSD"] < threshold:
                    success_names.append(item["name"])
            div = self.cal_diversity(set(success_names))
            overall[f"scRMSD_lt{threshold}"] = len(success_names) / len(results)
            overall[f"scRMSD_lt{threshold}_str"] = len(set(success_names)) / len(
                self.pdb_names
            )
            overall[f"div_scRMSD_lt{threshold}"] = div

        # scTM and scRMSD: max/min(all seq in a same design) -> avg over all designs
        overall_consistency = {}
        for item in results:
            key = item["name"]
            if key not in overall_consistency:
                overall_consistency[key] = {"scTM": 0.00001, "scRMSD": 10000.0}
            cur = overall_consistency[key]
            overall_consistency[key]["scTM"] = max(cur["scTM"], item["scTM"])
            overall_consistency[key]["scRMSD"] = min(cur["scRMSD"], item["scRMSD"])
        avg_tm = np.mean([v["scTM"] for v in overall_consistency.values()])
        avg_rmsd = np.mean([v["scRMSD"] for v in overall_consistency.values()])
        overall.update({"scTM": avg_tm, "scRMSD": avg_rmsd})
        sample_df = pd.DataFrame(results)
        sample_df = sample_df.sort_values(by=["name", "seq_idx"])
        summary_dict = {"task": self.task_type, "name": self.task_name}
        summary_dict.update(
            self.summary_from_df(
                sample_df,
                other_metrics=overall,
            )
        )
        sample_save_path, summary_save_path = save_eval_results(
            sample_df, summary_dict, self.out_dir, self.sample_fn, self.summary_fn
        )
        print(
            f"Eval done! Results are saved in {sample_save_path} and {summary_save_path}"
        )
        return {
            "task": self.task_type,
            "name": self.task_name,
            "sample_save_path": sample_save_path,
            "summary_save_path": summary_save_path,
        }
