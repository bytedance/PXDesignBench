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
from typing import Dict, List

from eval_design.tools.base import BasePredictor


class AF2ComplexPredictor(BasePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dir_name = os.path.dirname(__file__)
        self.script_path = os.path.join(dir_name, "main_af2_complex.py")

    def predict(
        self,
        input_dir: str,
        save_dir: str,
        data_list: List[Dict],
        cond_chain: str,
        binder_chain: str,
    ):
        input_data = {
            "input_dir": input_dir,
            "save_dir": save_dir,
            "data_list": data_list,
            "cond_chain": cond_chain,
            "binder_chain": binder_chain,
            "af2_cfg": self.cfg.to_dict(),
            "is_cyclic": self.cfg.get("is_cyclic", False),
        }
        output = self.run(input_data)
        for idx, item in enumerate(data_list):
            item.update(output[idx])
        return output


class AF2MonomerPredictor(BasePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dir_name = os.path.dirname(__file__)
        self.script_path = os.path.join(dir_name, "main_af2_monomer.py")

    def predict(
        self,
        save_dir: str,
        design_pdb_dir: str,
        data_list: List[Dict],
        binder_chain: str,
    ):
        input_data = {
            "save_dir": save_dir,
            "design_pdb_dir": design_pdb_dir,
            "data_list": data_list,
            "binder_chain": binder_chain,
            "af2_cfg": self.cfg.to_dict(),
            "is_cyclic": self.cfg.get("is_cyclic", False),
        }
        output = self.run(input_data)
        for idx, item in enumerate(data_list):
            item.update(output[idx])
        return output
