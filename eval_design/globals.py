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


def get_ckpt_path(version: str, file: str = ""):
    return f"/protenix/dataset/DesignCkpts:{version}/{file}"


TMALIGN_PATH = os.path.join(os.path.dirname(__file__), "metrics", "TMalign")

AF2_PARAMS_PATH = get_ckpt_path("af2")
ESMFOLD_MODEL_PATH = get_ckpt_path("esmfold")

MPNN_CKPT_PATH = {
    "ca": get_ckpt_path("mpnn", "ca_model_weights"),
    "bb": get_ckpt_path("mpnn", "vanilla_model_weights"),
    "soluble": get_ckpt_path("mpnn", "soluble_model_weights"),
}
