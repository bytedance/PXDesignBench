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

export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTENTION=true

use_deepspeed_evo_attention=true
dtype=bf16

input_dir="./examples/binder"
dump_dir="./output/binder"

binder_chains="B0"
is_mmcif=true
N_seqs=2
mpnn_temp=0.0001

python3 eval_design/run.py \
--data_dir ${input_dir} \
--dump_dir ${dump_dir} \
--is_mmcif ${is_mmcif} \
--binder.num_seqs ${N_seqs} \
--binder.tools.mpnn.temperature ${mpnn_temp} \
--binder.tools.af2.use_binder_template true \
--binder.tools.ptx.dtype ${dtype} \
--binder.tools.ptx.use_deepspeed_evo_attention ${use_deepspeed_evo_attention} \
--binder_chains ${binder_chains} \
--binder.use_gt_seq false 
