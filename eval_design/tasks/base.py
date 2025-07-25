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
import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from eval_design.metrics import diversity, secondary
from eval_design.tools.af2.af2_predictor import AF2ComplexPredictor, AF2MonomerPredictor
from eval_design.tools.ptx import ptx

logger = logging.getLogger(__name__)


class BaseTask(ABC):
    task_type: str
    task_name: str

    def __init__(self, input_data, cfg, device_id: int, seed: int):
        self.cfg = cfg
        self.device_id = device_id
        self.seed = seed
        self.pdb_dir = input_data["pdb_dir"]
        self.pdb_names = input_data["pdb_names"]
        self.out_dir = input_data.get("out_dir", os.path.dirname(self.pdb_dir))
        self.sample_fn = input_data.get("sample_fn", "sample_level_output.csv")
        self.summary_fn = input_data.get("summary_fn", "summary_output.json")

        # Default values
        self.num_seqs = cfg.get("num_seqs", 4)
        self.use_gt_seq = cfg.get("use_gt_seq", False)

        # Check pdb_paths
        self.process_pdb_paths()

    def get_device(self):
        if self.device_id >= 0:
            return f"cuda:{self.device_id}"
        else:
            return "cpu"

    @abstractmethod
    def design_sequence(self):
        pass

    @abstractmethod
    def run(self):
        """Execute the task"""
        pass

    @staticmethod
    def summary_from_df(
        sample_df: pd.DataFrame,
        exclude_keys=["name", "seq_idx", "sequence"],
        other_metrics={},
    ):
        metrics = {}
        for col in sample_df.columns:
            if col in exclude_keys:
                continue
            col_data = sample_df[col].dropna()

            # numeric column (float/int)
            if pd.api.types.is_numeric_dtype(sample_df[col]):
                values = col_data.values
                metrics[f"{col}.avg"] = float(np.mean(values))
                # metrics[f"{col}.std"] = float(np.std(values))

            # list of numbers (object column but with list values)
            elif pd.api.types.is_object_dtype(col_data):
                if all(isinstance(x, list) for x in col_data):
                    try:
                        # Flatten and compute avg/std per sample
                        per_sample_mean = col_data.apply(lambda x: np.mean(x))
                        metrics[f"{col}.avg"] = float(np.mean(per_sample_mean))
                        # metrics[f"{col}.std"] = float(np.std(per_sample_mean))
                    except:
                        pass  # fallback if something is not list of numbers

            if "_success" in col:
                metrics[f"{col}.count"] = int(np.sum(sample_df[col].astype(bool)))

        metrics.update(other_metrics)
        return metrics

    @staticmethod
    def compute_success_rate(filters_cfg, metrics: pd.DataFrame) -> pd.DataFrame:
        for filter_name, filter_details in filters_cfg.items():
            missing = [k for k in filter_details.keys() if k not in metrics.columns]

            def row_success(row):
                for metric_name, (sym, thres) in filter_details.items():
                    if metric_name not in row:
                        continue
                    value = row[metric_name]
                    if value is None:
                        return None
                    if isinstance(value, list):
                        # check whether there is any sample pass the filter
                        value = min(value) if sym == "<" else max(value)
                    if sym == "<" and value >= thres:
                        return 0
                    if sym == ">" and value <= thres:
                        return 0
                return 1

            if missing:
                print(
                    f"Missing columns {missing} for filter '{filter_name}'. Available columns: {list(metrics.columns)}"
                )
                metrics[f"{filter_name}_success"] = None
                metrics[f"{filter_name}_success_ignore_missing"] = metrics.apply(
                    row_success, axis=1
                )
                # DEBUG ONLY:
                # metrics[f"{filter_name}_success_ignore_missing"] = (
                #     np.random.rand(len(metrics)) < 0.1
                # )
            else:
                metrics[f"{filter_name}_success"] = metrics.apply(row_success, axis=1)
                metrics[f"{filter_name}_success_ignore_missing"] = metrics[
                    f"{filter_name}_success"
                ]
        return metrics

    def process_pdb_paths(self):
        "Temperary check for input pdb_paths"

        # Check if pdb_paths are valid
        valid_pdb_names = []
        for name in self.pdb_names:
            pdb_path = os.path.join(self.pdb_dir, name + ".pdb")
            # File exists
            if not os.path.exists(pdb_path):
                logger.warning(
                    f"pdb_path {pdb_path} does not exist. Will skip this file."
                )
                continue
            valid_pdb_names.append(name)
        self.pdb_names = list(valid_pdb_names)
        return

    def cal_diversity(self, pdb_names=None, binder_chain=None):
        # CPU eval, very slow
        if self.eval_diversity:
            all_names = self.pdb_names if pdb_names is None else pdb_names
            pdb_paths = [
                os.path.join(self.pdb_dir, name + ".pdb") for name in all_names
            ]
            div = diversity.compute_diversity(pdb_paths, binder_chain)
        else:
            div = -1
        return div

    def cal_secondary(self, results, chain_id=None):
        for item in results:
            pdb_path = os.path.join(self.pdb_dir, item["name"] + ".pdb")
            alpha, beta, loop = secondary.cacl_secondary_structure(pdb_path, chain_id)
            Rg, ref_ratio = secondary.get_chain_rg(pdb_path, chain_id)
            item.update(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "loop": loop,
                    "Rg": Rg,
                    "ref_ratio": ref_ratio,
                }
            )
            # print(item)

    def af2_complex_predict(self, data_list, save_dir, verbose=True):
        assert self.task_type in ["binder"]
        predictor = AF2ComplexPredictor(
            self.cfg.tools.af2,
            device_id=self.device_id,
            verbose=verbose,
            seed=self.seed,
        )
        predictor.predict(
            input_dir=self.pdb_dir,
            save_dir=save_dir,
            data_list=data_list,
            cond_chain=",".join(self.cond_chains),
            binder_chain=",".join(self.binder_chains),
        )

    def af2_monomer_predict(self, data_list, save_dir, verbose=True):
        assert self.task_type in ["binder", "ligand_binder"]
        predictor = AF2MonomerPredictor(
            self.cfg.tools.af2,
            device_id=self.device_id,
            verbose=verbose,
            seed=self.seed,
        )
        predictor.predict(
            save_dir=save_dir,
            design_pdb_dir=self.pdb_dir,
            data_list=data_list,
            binder_chain=self.binder_chains[0],
        )

    def protenix_predict(self, data_list, is_large=False):
        if is_large:
            ptx_cfg = self.cfg.tools.ptx_large
            dump_dir = os.path.join(self.out_dir, "ptx_pred_large")
        else:
            ptx_cfg = self.cfg.tools.ptx
            dump_dir = os.path.join(self.out_dir, "ptx_pred")
        ptx_filter = ptx.ProtenixFilter(cfg=ptx_cfg, device=self.get_device())
        # HARDCODE binder chain idx
        binder_chain_idx = 0 if self.binder_chains[0] == "A" else None
        json_path = ptx_filter.prepare_json(
            self.pdb_dir,
            data_list,
            dump_dir=dump_dir,
            binder_chain_idx=binder_chain_idx,
        )
        pred_pdb_paths = ptx_filter.predict(
            input_json_path=json_path,
            design_pdb_dir=self.pdb_dir,
            data_list=data_list,
            dump_dir=dump_dir,
            seed=self.seed,
            N_sample=ptx_cfg.N_sample,
            N_step=ptx_cfg.N_step,
            step_scale_eta=ptx_cfg.step_scale_eta,
            gamma0=ptx_cfg.gamma0,
            N_cycle=ptx_cfg.N_cycle,
            binder_chain_idx=binder_chain_idx,
            suffix="_large" if is_large else "",
        )
        return pred_pdb_paths
