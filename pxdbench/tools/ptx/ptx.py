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
import tempfile
from contextlib import nullcontext
from copy import deepcopy
from glob import glob
from typing import Any, Mapping

import numpy as np
import torch
from protenix.data.infer_data_pipeline import InferenceDataset
from protenix.data.json_maker import cif_to_input_json
from protenix.data.utils import pdb_to_cif
from protenix.model.protenix import Protenix
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import to_device
from runner.dumper import DataDumper

from pxdbench.permutation import permute_generated_min_complex_rmsd
from pxdbench.tools.ptx.interface import ProtenixAPI
from pxdbench.tools.ptx.ptx_utils import (
    download_infercence_cache,
    get_configs,
    patch_with_orig_seqs,
    populate_msa_with_cache,
)
from pxdbench.utils import concat_dict_values, convert_cif_to_pdb

logger = logging.getLogger(__name__)


class ProtenixFilter(ProtenixAPI):
    def __init__(self, cfg, device="cuda:0"):
        self.cfg = cfg
        self.model_name = cfg.model_name
        self.ptx_cfg = get_configs(self.model_name)
        self.ptx_cfg.model_name = self.model_name
        self.ptx_cfg.use_deepspeed_evo_attention = self.cfg.get(
            "use_deepspeed_evo_attention", True
        )
        self.ptx_cfg.data.msa.min_size.test = 2000
        self.ptx_cfg.data.msa.sample_cutoff.test = 2000
        if self.cfg.get("load_checkpoint_dir", ""):
            self.ptx_cfg.load_checkpoint_dir = self.cfg.load_checkpoint_dir
        self.ptx_ckpt_path = f"{self.ptx_cfg.load_checkpoint_dir}/{self.model_name}.pt"
        self.dtype = cfg.dtype
        self.device = device
        self.init_model()

    def init_model(self):
        _, model_size, model_feature, model_version = self.model_name.split("_")
        logger.info(
            f"Inference by Protenix: model_size: {model_size}, with_feature: {model_feature.replace('-',', ')}, model_version: {model_version}"
        )
        download_infercence_cache(self.ptx_cfg)

        self.model = Protenix(self.ptx_cfg).to(self.device)
        print(f"Loading protenix filter model from {self.ptx_ckpt_path}, strict: True")
        checkpoint = torch.load(self.ptx_ckpt_path, self.device)
        sample_key = [k for k in checkpoint["model"].keys()][0]
        print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items()
            }
        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=True,
        )
        self.model.eval()

    @torch.no_grad()
    def predict_one(
        self, data: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, torch.Tensor]:
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.dtype]

        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )

        data = to_device(data, self.device)

        with enable_amp:
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
            )

        return prediction

    @staticmethod
    def prepare_json(
        input_dir: str,
        data_list: list[dict],
        dump_dir: str,
        binder_chain_idx=None,
        orig_seqs: list = None,
        use_template=False,
    ):
        input_dicts = []
        for item in data_list:
            name = item["name"]
            seq = item["sequence"]
            seq_idx = item["seq_idx"]

            pdb_path = os.path.join(input_dir, name + ".pdb")
            with tempfile.NamedTemporaryFile(suffix=".cif") as tmp:
                tmp_cif_file = tmp.name
                pdb_to_cif(pdb_path, tmp_cif_file)
                d = cif_to_input_json(
                    tmp_cif_file, sample_name=name, save_entity_and_asym_id=True
                )

            if binder_chain_idx is None:
                b_id = len(d["sequences"]) - 1
            else:
                b_id = binder_chain_idx

            new_d = deepcopy(d)
            new_d["sequences"][b_id]["proteinChain"]["sequence"] = seq
            new_d["sequences"][b_id]["proteinChain"]["use_msa"] = False
            new_d["name"] = d["name"] + f"_seq{seq_idx}"
            input_dicts.append(new_d)

        if orig_seqs is not None:
            # cause the input must be PDB file, we will trim the chain id
            input_dicts = patch_with_orig_seqs(
                input_dicts, orig_seqs, trim=True, use_template=use_template
            )

        # precompute MSA if necessary
        input_dicts = populate_msa_with_cache(input_dicts)

        os.makedirs(dump_dir, exist_ok=True)
        json_path = os.path.join(dump_dir, "protenix_inputs.json")
        with open(json_path, "w") as f:
            json.dump(input_dicts, f, indent=4)
        return json_path

    def make_is_cyclic_mask_feat(self, data):
        """
        Take the last chain as cyclic binder chain and assign is_cyclic_mask to the input_feature_dict.
        """
        data["input_feature_dict"]["is_cyclic_mask"] = torch.zeros_like(
            data["input_feature_dict"]["residue_index"]
        )
        asym_id = data["input_feature_dict"]["asym_id"]

        # assume the binder chain is the last chain
        data["input_feature_dict"]["is_cyclic_mask"] = asym_id == asym_id.max()
        return data

    def predict(
        self,
        input_json_path: str,
        design_pdb_dir: str,
        data_list: list[dict],
        dump_dir: str,
        seed=2025,
        N_sample=1,
        N_step=2,
        step_scale_eta=1.0,
        gamma0=0,
        N_cycle=4,
        verbose=True,
        binder_chain_idx=None,
        is_cyclic=False,
        use_msa=True,
        suffix="",
    ):
        inference_dataset = InferenceDataset(
            input_json_path=input_json_path,
            dump_dir=None,
            use_msa=use_msa,
            configs=self.ptx_cfg,
        )
        os.makedirs(dump_dir, exist_ok=True)
        dumper = DataDumper(base_dir=dump_dir)

        all_predictions = {}
        seed = seed if isinstance(seed, int) else 2025
        seed_everything(seed=seed, deterministic=False)
        self.model.configs.sample_diffusion["N_sample"] = N_sample
        self.model.configs.sample_diffusion["N_step"] = N_step
        self.model.configs.sample_diffusion["step_scale_eta"] = step_scale_eta
        self.model.configs.sample_diffusion["gamma0"] = gamma0
        self.model.N_cycle = N_cycle
        self.model.configs.model.N_cycle = N_cycle
        pred_pdb_paths = {}
        for idx in range(len(inference_dataset)):
            data, atom_array, data_error_message = inference_dataset[idx]
            if is_cyclic:
                data = self.make_is_cyclic_mask_feat(data)
            sample_name = data["sample_name"]
            save_dir = dumper._get_dump_dir("", sample_name, seed)
            if len(data_error_message) > 0:
                print(f"Skip data {idx} because of the error: {data_error_message}")
                continue

            print(
                (
                    f"[Rank ({data['sample_index'] + 1}/{len(inference_dataset)})] {sample_name}: "
                    f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                    f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
                )
            )
            prediction = self.predict_one(data)
            stats = prediction["summary_confidence"]
            dumper.dump(
                "",
                sample_name,
                seed,
                pred_dict=prediction,
                atom_array=atom_array,
                entity_poly_type=data["entity_poly_type"],
            )

            assert sample_name not in all_predictions
            # HARDCODE: now the last chain is the binder chain
            stat_list = []
            for sample_id in range(N_sample):
                s = stats[sample_id]
                # save pdb
                pred_cif_path = glob(
                    os.path.join(
                        save_dir,
                        "predictions",
                        f"{sample_name}_*sample_{sample_id}.cif",
                    )
                )
                assert len(pred_cif_path) == 1
                pred_cif_path = pred_cif_path[0]
                pred_pdb_path = pred_cif_path[:-4] + ".pdb"
                convert_cif_to_pdb(pred_cif_path, pred_pdb_path)
                if sample_id == 0:
                    # only save the first sample
                    # in the future, if the design model outputs both sequence and structure, we may not need re-docked complex as inputs anymore
                    pred_pdb_paths[sample_name] = pred_pdb_path

                # compute predict-design RMSD
                design_pdb_path = os.path.join(
                    design_pdb_dir, sample_name.rsplit("_seq", 1)[0] + ".pdb"
                )
                if os.path.isfile(design_pdb_path):
                    rmsd = permute_generated_min_complex_rmsd(
                        pred_pdb_path, design_pdb_path, pred_pdb_path
                    )
                else:
                    rmsd = None
                if rmsd is not None:
                    rmsd = round(rmsd, 2)

                if binder_chain_idx is None:
                    binder_chain_idx = len(s["chain_ptm"]) - 1
                target_chain_idx = [
                    c for c in range(len(s["chain_ptm"])) if c != binder_chain_idx
                ]
                ptm_target = [
                    (
                        s["chain_ptm"][b].item()
                        if torch.is_tensor(s["chain_ptm"])
                        else s["chain_ptm"][b]
                    )
                    for b in target_chain_idx
                ]

                ptx_s = {
                    f"ptx{suffix}_plddt": float(s["plddt"]),
                    f"ptx{suffix}_ptm_binder": float(s["chain_ptm"][binder_chain_idx]),
                    f"ptx{suffix}_ptm_target": np.mean(ptm_target),
                    f"ptx{suffix}_iptm": float(s["iptm"]),
                    f"ptx{suffix}_ptm": float(s["ptm"]),
                    f"ptx{suffix}_iptm_binder": float(
                        s["chain_iptm"][binder_chain_idx]
                    ),
                    f"ptx{suffix}_pred_design_rmsd": rmsd,
                }
                stat_list.append(ptx_s)

            stat = concat_dict_values(stat_list)

            # take mean value of N_sample predictions and round to 4 digits
            for k, v in stat.items():
                if v[0] is None:
                    stat[k] = None
                else:
                    stat[k] = round(sum(v) / len(v), 4)

            all_predictions[sample_name] = stat
            if verbose:
                print(f"{sample_name}, {stat}")

        for item in data_list:
            design_name = item["name"] + f"_seq{item['seq_idx']}"
            assert design_name in all_predictions
            item.update(all_predictions[design_name])

        return pred_pdb_paths

    def inference_only(
        self,
        input_json_path: str,
        dump_dir: str,
        seed=2025,
        N_sample=1,
        N_step=2,
        step_scale_eta=1.0,
        gamma0=0,
        N_cycle=4,
        use_msa=True,
    ):
        inference_dataset = InferenceDataset(
            input_json_path=input_json_path,
            use_msa=use_msa,
            dump_dir=None,
            configs=self.ptx_cfg,
        )
        os.makedirs(dump_dir, exist_ok=True)
        dumper = DataDumper(base_dir=dump_dir)

        seed = seed if isinstance(seed, int) else 2025
        seed_everything(seed=seed, deterministic=False)
        self.model.configs.sample_diffusion["N_sample"] = N_sample
        self.model.configs.sample_diffusion["N_step"] = N_step
        self.model.configs.sample_diffusion["step_scale_eta"] = step_scale_eta
        self.model.configs.sample_diffusion["gamma0"] = gamma0
        self.model.N_cycle = N_cycle
        self.model.configs.model.N_cycle = N_cycle
        pred_pdb_paths = {}
        pred_stats = {}
        for idx in range(len(inference_dataset)):
            data, atom_array, data_error_message = inference_dataset[idx]
            sample_name = data["sample_name"]
            save_dir = dumper._get_dump_dir("", sample_name, seed)
            if len(data_error_message) > 0:
                print(f"Skip data {idx} because of the error: {data_error_message}")
                continue

            print(
                (
                    f"[Rank ({data['sample_index'] + 1}/{len(inference_dataset)})] {sample_name}: "
                    f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                    f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
                )
            )
            prediction = self.predict_one(data)
            # keys: ['coordinate', 'summary_confidence', 'full_data', 'plddt', 'plddt_un', 'pae', 'pde', 'resolved'])
            stats = prediction["summary_confidence"]
            dumper.dump(
                "",
                sample_name,
                seed,
                pred_dict=prediction,
                atom_array=atom_array,
                entity_poly_type=data["entity_poly_type"],
            )
            pred_cif_path = os.path.join(
                save_dir,
                "predictions",
                f"{sample_name}_seed_{seed}_sample_0.cif",
            )
            pred_pdb_path = os.path.join(dump_dir, f"{sample_name}.pdb")
            convert_cif_to_pdb(pred_cif_path, pred_pdb_path)
            pred_pdb_paths[sample_name] = pred_pdb_path
            pred_stats[sample_name] = stats
        return pred_pdb_paths, pred_stats
