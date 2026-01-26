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
import tempfile
from typing import Any, Dict, List

from pxdbench.globals import MPNN_CKPT_PATH, _require


class VanillaMPNNPredictor:
    def __init__(self, cfg, device_id: int = 0, seed: int = None, verbose=True):
        self.cfg = cfg
        self.device_id = device_id
        self.seed = seed
        self.verbose = verbose
        _require(
            os.path.join(
                MPNN_CKPT_PATH[self.cfg["model_type"]], self.cfg["model_name"] + ".pt"
            )
        )

        dir_name = os.path.dirname(__file__)
        self.prepare_script_path = os.path.join(dir_name, "parse_multiple_chains.py")
        self.run_script_path = os.path.join(dir_name, "protein_mpnn_run.py")
        self.env = os.environ.copy()
        self.env["CUDA_VISIBLE_DEVICES"] = str(device_id)

    def prepare_jsonl(self, input_dir: str, pdb_names: list[str]):
        """
        Prepare a JSONL file for ProteinMPNN input by parsing multiple chains from PDB files.

        Args:
            input_dir (str): Directory containing input PDB files.
            pdb_names (list[str]): List of PDB base names (without '.pdb') to process.

        Returns:
            str: Path to the output JSONL file.
        """

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name

        output_path = os.path.join(input_dir, "parsed_pdbs.jsonl")
        cmd = [
            "python3",
            "-u",
            self.prepare_script_path,
            "--input_path",
            input_dir,
            "--output",
            output_path,
        ]

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
            print(f"Run subprocess success: {returncode}")

        except Exception as e:
            print(f"Run subprocess fail: {str(e)}")
            raise

        result = []
        with open(output_path, "r") as infile:
            for line in infile:
                entry = json.loads(line)
                if entry.get("name") in pdb_names:
                    result.append(entry)
        with open(output_path, "w") as f:
            for entry in result:
                f.write(json.dumps(entry) + "\n")
        return output_path

    def run_mpnn(self, jsonl_path: str, pdb_names: list[str], num_seqs: int):
        """
        Run ProteinMPNN to design sequences for the given PDB names.

        Args:
            jsonl_path (str): Path to the input JSONL file containing parsed PDB data.
            pdb_names (list[str]): List of PDB base names (without '.pdb') to process.
            num_seqs (int): Number of sequences to generate per PDB.

        Returns:
            None
        """

        model_type = self.cfg["model_type"]
        model_name = self.cfg["model_name"]
        assert model_type in ["ca", "bb", "soluable"]
        path_to_model_weights = MPNN_CKPT_PATH[model_type]

        output_dir = os.path.dirname(jsonl_path)
        cmd = [
            "python3",
            "-u",
            self.run_script_path,
            "--jsonl_path",
            jsonl_path,
            "--out_folder",
            output_dir,
            "--num_seq_per_target",
            str(num_seqs),
            "--sampling_temp",
            self.cfg["temperature"],
            "--batch_size",
            str(num_seqs),
            "--model_name",
            model_name,
            "--path_to_model_weights",
            path_to_model_weights,
        ]
        bias_aa_json = self.cfg.get("bias_aa_json", "")
        if bias_aa_json:
            cmd.extend(["--bias_AA_jsonl", bias_aa_json])
        if model_type == "ca":
            cmd.extend(["--ca_only"])
        elif model_type == "soluble":
            cmd.extend(["--use_soluble_model"])
        if self.seed is not None:
            cmd.extend(["--seed", str(self.seed)])

        print("Run Protein MPNN with %s" % (" ".join(cmd)))
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
            print(f"Run subprocess success: {returncode}")

        except Exception as e:
            print(f"Run subprocess fail: {str(e)}")
            raise

        # postprocess
        seq_dir = os.path.join(output_dir, "seqs")
        result = []
        for name in pdb_names:
            fasta_file = name + ".fa"
            fasta_path = os.path.join(seq_dir, fasta_file)
            if not os.path.exists(fasta_path):
                continue

            with open(fasta_path, "r") as f:
                lines = f.readlines()

            for i, line_no in enumerate(
                range(4, int(num_seqs) * 2 + 3, 2)
            ):  # lines 4,6,...,18 (1-based)
                if line_no <= len(lines):
                    sequence = lines[line_no - 1].strip()
                    result.append({"name": name, "seq_idx": i, "sequence": sequence})
        return result

    def design_monomer(
        self, pdb_dir: str, pdb_names: List[str], num_samples: int
    ) -> List[Dict]:
        """
        Design sequences for monomer proteins using ProteinMPNN.

        Args:
            pdb_dir (str): Directory containing input PDB files.
            pdb_names (list[str]): List of PDB base names (without '.pdb') to process.
            num_samples (int): Number of sequences to generate per PDB.

        Returns:
            list[dict]: List of design results with keys 'name' (PDB name), 'seq_idx' (sequence index),
                        and 'sequence' (designed amino acid sequence).
        """

        # input_data = {
        #     "pdb_dir": pdb_dir,
        #     "pdb_names": pdb_names,
        #     "num_samples": num_samples,
        #     "mpnn_cfg": self.cfg.to_dict(),
        #     "design_type": "monomer",
        # }
        jsonl_path = self.prepare_jsonl(pdb_dir, pdb_names)
        output = self.run_mpnn(jsonl_path, pdb_names, num_samples)
        os.unlink(jsonl_path)
        return output

    def design_binder(
        self,
        pdb_dir: str,
        pdb_names: List[str],
        num_samples: int,
        binder_chains: List[str],
        cond_chains: List[str],
    ) -> List[Dict]:
        """
        Design sequences for binder proteins using ProteinMPNN.

        Args:
            pdb_dir (str): Directory containing input PDB files.
            pdb_names (list[str]): List of PDB base names (without '.pdb') to process.
            num_samples (int): Number of sequences to generate per PDB.
            binder_chains (list[str]): List of binder chain IDs.
            cond_chains (list[str]): List of conditional chain IDs.

        Returns:
            list[dict]: List of design results with keys 'name' (PDB name), 'seq_idx' (sequence index),
                        and 'sequence' (designed amino acid sequence).
        """
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
