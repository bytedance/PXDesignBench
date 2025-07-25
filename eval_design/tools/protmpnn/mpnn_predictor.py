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
from typing import Any, Dict, List

from eval_design.tools.base import BasePredictor


class MPNNPredictor(BasePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dir_name = os.path.dirname(__file__)
        self.script_path = os.path.join(dir_name, "main_mpnn.py")

    def design_monomer(
        self, pdb_dir: str, pdb_names: List[str], num_samples: int
    ) -> List[Dict]:

        input_data = {
            "pdb_dir": pdb_dir,
            "pdb_names": pdb_names,
            "num_samples": num_samples,
            "mpnn_cfg": self.cfg.to_dict(),
            "design_type": "monomer",
        }
        output = self.run(input_data)
        return output

    def design_binder(
        self,
        pdb_dir: str,
        pdb_names: List[str],
        num_samples: int,
        binder_chains: List[str],
        cond_chains: List[str],
    ) -> List[Dict]:
        input_data = {
            "pdb_dir": pdb_dir,
            "pdb_names": pdb_names,
            "num_samples": num_samples,
            "binder_chains": binder_chains,
            "cond_chains": cond_chains,
            "mpnn_cfg": self.cfg.to_dict(),
            "design_type": "binder",
        }
        output = self.run(input_data)
        return output
