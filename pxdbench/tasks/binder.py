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

import json
import os

import pandas as pd

from pxdbench.tasks.base import BaseTask
from pxdbench.tools.protmpnn.main_mpnn import get_gt_sequence
from pxdbench.tools.protmpnn.mpnn_predictor import MPNNPredictor
from pxdbench.utils import save_eval_results

from .registry import register_task


@register_task("binder")
class BinderTask(BaseTask):
    def __init__(self, input_data, cfg, device_id: int, seed: int):
        """
        Initialize a BinderTask instance.

        Args:
            input_data (dict): Task input parameters including PDB paths and chain specifications.
            cfg (dict): Configuration dictionary with task settings.
            device_id (int): GPU device ID (-1 for CPU).
            seed (int): Random seed for reproducibility.

        Validates:
            - Exactly one binder chain is specified (multiple binder chains not supported).
        """
        self.task_type = "binder"
        self.task_name = input_data.get("name", "binder")
        assert "cond_chains" in input_data
        assert "binder_chains" in input_data
        self.cond_chains = input_data["cond_chains"]
        self.binder_chains = input_data["binder_chains"]
        self.pdb_name_to_binder_seq_list = input_data.get(
            "pdb_name_to_binder_seq_list", None
        )
        if input_data.get("orig_seqs_json", None) is not None:
            with open(input_data["orig_seqs_json"], "r") as f:
                self.orig_seqs = json.load(f)
        elif input_data.get("orig_seqs", None) is not None:
            self.orig_seqs = input_data["orig_seqs"]
        else:
            self.orig_seqs = None

        # Default values
        self.use_binder_seq_list = cfg.get("use_binder_seq_list", False)
        self.eval_diversity = cfg.get("eval_diversity", False)
        self.eval_binder_monomer = cfg.get("eval_binder_monomer", True)
        self.eval_complex = cfg.get("eval_complex", True)
        self.eval_protenix_mini = cfg.get("eval_protenix_mini", True)
        self.eval_protenix = cfg.get("eval_protenix", False)

        # Check values
        assert (
            len(self.binder_chains) == 1
        ), f"Get {len(self.binder_chains)} binder chains, but only 1 is allowed."

        super().__init__(input_data, cfg, device_id, seed)

    def prepare_data_from_seq_list(self):
        datas = []
        for name in self.pdb_names:
            binder_seq_list = self.pdb_name_to_binder_seq_list[name]
            for i, seq in enumerate(binder_seq_list):
                data = {"name": name, "seq_idx": i, "sequence": seq}
                datas.append(data)
        return datas

    def design_sequence(self, verbose=True):
        """
        Generates binder sequences based on task configuration.

        Supports three modes:
        1. Use pre-provided sequence lists (self.use_binder_seq_list)
        2. Use ground truth sequences from PDB files (self.use_gt_seq)
        3. De novo design using MPNN (default)

        Args:
            verbose (bool, optional): Whether to print detailed progress. Defaults to True.

        Returns:
            list[dict]: List of design results with keys "name", "seq_idx", and "sequence".
        """
        if self.use_binder_seq_list:
            results = self.prepare_data_from_seq_list()
        elif self.use_gt_seq:
            results = get_gt_sequence(
                self.pdb_dir, self.pdb_names, self.binder_chains[0]
            )
        else:
            mpnn_predictor = MPNNPredictor(
                self.cfg.tools.mpnn,
                device_id=self.device_id,
                verbose=verbose,
                seed=self.seed,
            )
            results = mpnn_predictor.design_binder(
                self.pdb_dir,
                self.pdb_names,
                self.num_seqs,
                binder_chains=self.binder_chains,
                cond_chains=self.cond_chains,
            )
        return results

    def run(self):
        """
        Executes the complete binder design evaluation workflow.

        Workflow steps:
        1. Designs sequences via design_sequence()
        2. Runs structure predictions (AF2 complex/monomer, Protenix) based on config
        3. Calculates secondary structure and diversity metrics
        4. Saves sample-level results to CSV and summary metrics to JSON

        Returns:
            dict: Dictionary with task metadata and output file paths.
        """
        results = self.design_sequence()
        self.check_results(results)
        binder_chain = self.binder_chains[0]

        af2_pred_path = os.path.join(self.out_dir, "af2_pred")
        if self.eval_complex:
            self.af2_complex_predict(results, af2_pred_path)

        if self.eval_binder_monomer:
            self.af2_monomer_predict(results, af2_pred_path)

        if self.eval_protenix_mini:
            self.protenix_predict(results, orig_seqs=self.orig_seqs)

        if self.eval_protenix:
            self.protenix_predict(results, orig_seqs=self.orig_seqs, is_large=True)

        self.cal_secondary(results, binder_chain)
        div = self.cal_diversity()
        sample_df = pd.DataFrame(results)
        sample_df = sample_df.sort_values(by=["name", "seq_idx"])
        self.compute_success_rate(self.cfg.filters, sample_df)
        summary_dict = {"task": self.task_type, "name": self.task_name}
        summary_dict.update(
            self.summary_from_df(sample_df, other_metrics={"diversity": div})
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

    def check_results(self, results):
        """
        Validates design results for consistency and correctness.

        Checks:
        1. No duplicate entries (by structure name + sequence index)
        2. Correct number of sequences per structure (when not using pre-provided lists)

        Args:
            results (list[dict]): List of design results from design_sequence()

        Raises:
            ValueError: If duplicates are found or sequence count is incorrect.
        """
        result_names = [
            result["name"] + f"_seq{result['seq_idx']}" for result in results
        ]
        if len(result_names) != len(set(result_names)):
            raise ValueError(f"Found duplicate names in results: {result_names}.")
        if self.use_binder_seq_list or self.use_gt_seq:
            pass
        elif len(result_names) != len(self.pdb_names) * self.num_seqs:
            raise ValueError(
                f"Found {len(result_names)} results, but {len(self.pdb_names)} pdb_names, each with {self.num_seqs} seqs are provided."
            )
        return
