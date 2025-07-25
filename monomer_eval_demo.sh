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

input_dir="./examples/monomer"
dump_dir="./output/monomer"

is_mmcif=false
N_seqs=8
mpnn_temp=0.1
mpnn_model=ca

python3 ./eval_design/run_monomer.py \
--data_dir ${input_dir} \
--dump_dir ${dump_dir} \
--is_mmcif ${is_mmcif} \
--monomer.num_seqs ${N_seqs} \
--monomer.tools.mpnn.temperature ${mpnn_temp} \
--monomer.tools.mpnn.model_type ${mpnn_model} \
--monomer.eval_diversity false
