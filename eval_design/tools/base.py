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
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod


class BasePredictor(ABC):
    def __init__(self, cfg, device_id: int = 0, seed: int = None, verbose=True):
        self.cfg = cfg
        self.device_id = device_id
        self.seed = seed
        self.verbose = verbose
        self.model_loaded = False
        self.process = None

        self.env = os.environ.copy()
        self.env["CUDA_VISIBLE_DEVICES"] = str(device_id)
        self.env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
        self.env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    def run(self, input_data):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(input_data, f)
            input_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        cmd = [
            "python3",
            "-u",
            self.script_path,
            "--input",
            input_path,
            "--output",
            output_path,
        ]
        if self.seed is not None:
            cmd.extend(["--seed", str(self.seed)])

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.env,
            )
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())

            while True:
                error = process.stderr.readline()
                if error == "" and process.poll() is not None:
                    break
                if error:
                    print(error.strip())

            returncode = process.wait()
            if self.verbose:
                print(f"Run subprocess success: {returncode}")

            with open(output_path, "r") as f:
                return json.load(f)

        except Exception as e:
            print(f"Run subprocess fail: {str(e)}")
            raise

        finally:
            # clean temp files
            os.unlink(input_path)
            os.unlink(output_path)
