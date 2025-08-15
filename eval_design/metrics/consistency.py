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

from eval_design.metrics import Kalign, tmalign


def self_consistency(inputs: dict):
    """
    Calculate self-consistency metrics (scTM and scRMSD) for a set of input files.

    Args:
        inputs (dict): A dictionary where keys are names and values are dictionaries
                       containing "source_file" and "target_file" keys.

    Returns:
        dict: A dictionary containing the calculated scTM and scRMSD metrics for each input.
    """
    outputs = {}
    for name, item in inputs.items():
        source_file = item["source_file"]
        target_file = item["target_file"]
        TMscore, rmsd = None, None
        if not os.path.exists(target_file):
            print(f"{target_file} does not exist!")
            outputs[name] = {"scTM": None, "scRMSD": None}
            continue

        # Run TMalign
        TMscore = tmalign.get_tm_score(source_file, target_file)

        # Run Kalign.py
        try:
            rmsd = Kalign.align_and_calculate_rmsd(source_file, target_file)
            rmsd = round(rmsd, 3)
        except Exception as e:
            print(f"Error running Kalign for {source_file} and {target_file}: {e}")
            rmsd = None
        outputs[name] = {"scTM": TMscore, "scRMSD": rmsd}
    return outputs
