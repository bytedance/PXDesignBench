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

import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb

from pxdbench.globals import ESMFOLD_MODEL_PATH, _require


class ESMFold:
    """
    Wrapper class for protein structure prediction using the ESMFold model.

    Handles model initialization, sequence tokenization, structure prediction,
    and conversion of model outputs to PDB format with pLDDT scores.
    """

    def __init__(self, device="cuda:0"):
        _require(os.path.join(ESMFOLD_MODEL_PATH, "config.json"))
        _require(os.path.join(ESMFOLD_MODEL_PATH, "pytorch_model.bin"))
        self.tokenizer = AutoTokenizer.from_pretrained(ESMFOLD_MODEL_PATH)
        self.model = EsmForProteinFolding.from_pretrained(
            ESMFOLD_MODEL_PATH,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.to(device)
        self.device = device

    def convert_outputs_to_pdb(self, outputs):
        final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
        outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = outputs["atom37_atom_exists"]
        pdbs = []
        pred_positions = []
        for i in range(outputs["aatype"].shape[0]):
            aa = outputs["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = outputs["residue_index"][i] + 1
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=outputs["plddt"][i],
                chain_index=(
                    outputs["chain_index"][i] if "chain_index" in outputs else None
                ),
            )
            pdbs.append(to_pdb(pred))
            pred_positions.append(outputs["positions"][-1][i])

        return pdbs, pred_positions

    def parse_plddt(self, plddt):
        num = plddt.shape[0]
        plddt_lis = []
        for i in range(num):
            one_plddt = torch.mean(plddt[i]).item()
            plddt_lis.append(one_plddt)
        return plddt_lis

    def predict(self, sequences):
        tokenized_input = self.tokenizer(
            sequences, return_tensors="pt", padding=True, add_special_tokens=False
        )["input_ids"]
        tokenized_input = tokenized_input.to(self.device)
        self.model.trunk.set_chunk_size(128)
        self.model.eval()
        with torch.no_grad():
            output = self.model(tokenized_input)
            pdbs, pred_position = self.convert_outputs_to_pdb(output)
            plddt = output["plddt"]
            plddt = self.parse_plddt(plddt)
        return pdbs, plddt
