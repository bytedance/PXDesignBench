# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under Creative Commons Attribution-NonCommercial 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://creativecommons.org/licenses/by-nc/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from natsort import natsorted
from protenix.config import parse_configs, parse_sys_args
from protenix.config.extend_types import RequiredValue
from protenix.utils.distributed import DIST_WRAPPER

from pxdbench.pxd_configs.eval import eval_configs
from pxdbench.run import find_files_with_ext, run_task
from pxdbench.utils import convert_cifs_to_pdbs

logger = logging.getLogger(__name__)


def scan_tasks(data_dir, is_mmcif=False):
    data_dir = Path(data_dir).resolve()
    paths = []

    pattern = "*.cif" if is_mmcif else "*.pdb"
    for pdb_file in data_dir.rglob(pattern):
        paths.append(os.path.dirname(pdb_file))

    return sorted(list(set(paths)))


class EvalRunner(object):
    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.root_dir = self.configs.data_dir
        self.dump_dir = self.configs.dump_dir
        self.init_env()

    def init_env(self) -> None:
        self.print(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            + f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if DIST_WRAPPER.world_size > 1:
            dist.init_process_group(backend="nccl")
        logging.info("Finished init ENV.")

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)

    def run(self):
        input_dirs = scan_tasks(self.root_dir, self.configs.is_mmcif)
        valid_input_dirs = []
        for data_dir in input_dirs:
            exp_name = os.path.relpath(data_dir, self.root_dir)
            if os.path.exists(
                os.path.join(self.dump_dir, exp_name, "summary_output.json")
            ):
                self.print(f"Found summary file for {exp_name} - Skip!")
                continue
            valid_input_dirs.append(data_dir)

        valid_input_dirs = natsorted(valid_input_dirs)
        logging.info(f"There are {len(valid_input_dirs)} tasks to evaluate")
        print(valid_input_dirs)

        sub_input_dirs = valid_input_dirs[DIST_WRAPPER.rank :: DIST_WRAPPER.world_size]
        for i, data_dir in enumerate(sub_input_dirs):
            logging.info(
                f"Begin to evaluate [{i + 1}/{len(sub_input_dirs)}]: {data_dir}"
            )
            exp_name = os.path.relpath(data_dir, self.root_dir)
            dump_dir = os.path.join(self.dump_dir, exp_name)
            os.makedirs(dump_dir, exist_ok=True)
            if self.configs.is_mmcif:
                pdb_dir, pdb_names, _, _ = convert_cifs_to_pdbs(
                    data_dir,
                    out_pdb_dir=os.path.join(data_dir, "converted_pdbs"),
                )
            else:
                pdb_dir = data_dir
                pdb_names = find_files_with_ext(data_dir, "pdb")
            logging.info(f"There are {len(pdb_names)} pdbs in this task.")

            input_data = {
                "task": "monomer",
                "name": exp_name,
                "pdb_dir": pdb_dir,
                "pdb_names": pdb_names,
                "out_dir": dump_dir,
            }
            run_task(
                input_data,
                self.configs,
                device_id=DIST_WRAPPER.local_rank,
                seed=self.configs.seed,
            )
        logging.info("Eval done!")


def main():
    config_dict = {
        "data_dir": RequiredValue(str),
        "dump_dir": RequiredValue(str),
        "is_mmcif": False,
        "seed": 2025,
        **eval_configs,
    }
    configs = parse_configs(config_dict, arg_str=parse_sys_args())
    runner = EvalRunner(configs)
    runner.run()


if __name__ == "__main__":
    main()
