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

# pylint: disable=C0114,C0301

from protenix.config.extend_types import ListValue

eval_configs = {
    # "input_dir": "",
    # "save_dir": DefaultNoneWithType(str),
    "monomer": {
        "eval_diversity": False,
        "num_seqs": 8,
        "tools": {
            "mpnn": {
                "model_type": "ca",  # [ca, bb, soluble]
                "model_name": "v_48_020",
                "temperature": "0.1",
            }
        },
    },
    "binder": {
        "eval_diversity": False,
        "eval_binder_monomer": True,
        "eval_complex": True,
        "eval_protenix": True,
        "eval_protenix_large": False,
        "num_seqs": 1,
        "use_gt_seq": False,
        "use_binder_seq_list": False,
        "is_cyclic": False,
        "tools": {
            "mpnn": {
                "weights": "original",  # [original, soluble]
                "rm_aa": "C",
                "temperature": "0.0001",
                "fix_interface": False,  # whether fixing interface restype or not
            },
            "af2": {
                "use_multimer": False,
                "model_ids": ListValue([0]),
                "use_initial_guess": True,
                "use_initial_atom_pos": False,
                "use_binder_template": True,
                "is_cyclic": False,
            },
            "ptx": {
                "model_name": "protenix_mini_default_v0.5.0",
                "dtype": "bf16",
                "use_deepspeed_evo_attention": True,
                "N_cycle": 4,
                "N_sample": 1,
                "N_step": 2,
                "step_scale_eta": 1.0,
                "gamma0": 0,
            },
            "ptx_large": {
                "model_name": "protenix_base_default_v0.5.0",
                "dtype": "bf16",
                "use_deepspeed_evo_attention": True,
                "N_cycle": 4,
                "N_sample": 1,
                "N_step": 2,
                "step_scale_eta": 1.0,
                "gamma0": 0,
            },
        },
        "filters": {
            "bindcraft": {
                "pLDDT": (">", 0.8),
                "i_pTM": (">", 0.5),
                "i_pAE": ("<", 0.35),
                "bound_unbound_RMSD": ("<", 3.5),
            },
            "af2_opt": {
                "pLDDT": (">", 0.9),
                "unscaled_i_pAE": ("<", 7.0),
                "af2_binder_pred_design_rmsd": ("<", 1.5),
            },
            "ptx_mini": {
                "ptx_iptm_binder": (">", 0.85),
                "ptx_ptm_binder": (">", 0.88),
                "ptx_pred_design_rmsd": ("<", 2.5),
            },
            "ptx_large": {
                "ptx_large_iptm_binder": (">", 0.85),
                "ptx_large_ptm_binder": (">", 0.88),
                "ptx_large_pred_design_rmsd": ("<", 2.5),
            },
        },
    },
}
