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
  # ---- AF2 ----
  "$ROOT/af2/params_model_1.npz"
  "$ROOT/af2/params_model_1_ptm.npz"
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
input_dir="./examples/binder"
dump_dir="./output/binder"

binder_chains="B0"
is_mmcif=true
N_seqs=2
mpnn_temp=0.0001

python3 ./pxdbench/run.py \
--data_dir ${input_dir} \
--dump_dir ${dump_dir} \
--is_mmcif ${is_mmcif} \
--seed 2025 \
--orig_seqs_json ./examples/orig_seqs_test.json \
--binder.num_seqs ${N_seqs} \
--binder.tools.mpnn.temperature ${mpnn_temp} \
--binder.tools.af2.use_binder_template true \
--binder.tools.ptx_mini.dtype ${dtype} \
--binder.tools.ptx_mini.use_deepspeed_evo_attention ${use_deepspeed_evo_attention} \
--binder_chains ${binder_chains} \
--binder.use_gt_seq false 
