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

dtype=fp32
use_deepspeed_evo_attention=false

export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTENTION=${use_deepspeed_evo_attention}
export TOOL_WEIGHTS_ROOT="$(pwd)/tool_weights"

# ===============================
# Tool Weights Sanity Check
# ===============================
ROOT="${TOOL_WEIGHTS_ROOT}"
declare -a REQUIRED_FILES=(
  # ---- ESMFold ----
  "$ROOT/esmfold/pytorch_model.bin"
)
echo "Checking tool weights in: $ROOT"
for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo -e "\nMissing required tool weight:"
    echo "   $f"
    echo -e "\nPlease run:"
    echo "   bash download_tool_weights.sh"
    exit 1
  fi
done

# ===============================
# Main
# ===============================
input_dir="./examples/monomer"
dump_dir="./output/monomer"

is_mmcif=false
N_seqs=8
mpnn_temp=0.1
mpnn_model=ca

python3 ./pxdbench/run_monomer.py \
--data_dir ${input_dir} \
--dump_dir ${dump_dir} \
--is_mmcif ${is_mmcif} \
--seed 2025 \
--monomer.num_seqs ${N_seqs} \
--monomer.tools.mpnn.temperature ${mpnn_temp} \
--monomer.tools.mpnn.model_type ${mpnn_model} \
--monomer.eval_diversity false
