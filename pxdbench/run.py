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
from typing import Union

import torch
from protenix.config import parse_configs, parse_sys_args
from protenix.config.extend_types import ListValue, RequiredValue
from protenix.utils.logger import get_logger

from pxdbench.pxd_configs.eval import eval_configs
from pxdbench.tasks import get_task_class
from pxdbench.utils import convert_cif_to_pdb

logger = get_logger(__name__)


def run_task(input_data: dict, configs, device_id: int = 0, seed: int = None):
    """

    Args:
        input_data (dict): A dictionary containing the following keys:
            task (str): The task to run. Example: "binder".
            pdb_dir (str): The directory containing the PDB files.
            pdb_names (list): The names of the PDB files.
            cond_chains (list): The chains to condition on.
            binder_chains (list): The chains to bind.
            out_dir (str): The root directory. Outputs will be saved here.
        configs (dict): A dictionary containing the configuration for the task.
        device_id (int): Device ID to run the task on.
        seed (int): Random seed to use.
    Returns:
        dict: A dictionary containing the results of the task.
    """
    if device_id >= 0 and not torch.cuda.is_available():
        raise ValueError("device_id must be -1 (CPU) or a valid GPU ID")
    task = input_data["task"]
    task_cls = get_task_class(task)
    task_cfg = configs.get(task)
    task = task_cls(input_data, task_cfg, device_id, seed)
    return task.run()


def split_string(s):
    return s.split(",")


def get_file_name_list(file_name_list: Union[str, list]):
    """
    Get file names from a string or a file.
    Args:
        file_name_list (str or list): A string or a list containing file names.
            If a string, it should be a comma-separated list of file names.
            If a file, it should contain one file name per line.
    Returns:
        list: A list of file names.
    """
    if isinstance(file_name_list, list):
        return file_name_list
    if os.path.exists(file_name_list):
        file_names = []
        with open(file_name_list, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    file_names.append(line)
        return file_names
    else:
        file_name_list = file_name_list.split(",")
        return file_name_list


def find_files_with_ext(folder_path, ext="cif"):
    pdb_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(f".{ext}"):
            pdb_files.append(filename[: -(len(ext) + 1)])
    return sorted(pdb_files)


def get_chains_from_pdb(pdb_path):
    chains = set()
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21].strip()
                if chain_id:
                    chains.add(chain_id)
    return chains


def prepare_tasks_from_mmcif(
    data_dir: str,
    file_name_list: list,
    binder_chains: list[str],
    cond_chains: list[str],
    dump_dir: str,
):
    """
    Prepare tasks from mmCIF files.
    Args:
        data_dir (str): The directory containing the mmCIF files.
        file_name_list (str or list): A string or a list containing file names.
            If a string, it should be a comma-separated list of file names.
            If a file, it should contain one file name per line.
        binder_chains (list): The binder chain IDs.
        cond_chains (list): The chains to condition on. Could be auto infered by binder_chains.
        root_dir (str): The root directory. Outputs will be saved here.
    Returns:
        dict: A dictionary containing the results of the task.
    """
    n_cond_chains, n_binder_chains = None, None

    valid_file_name_list = []
    for file_name in file_name_list:
        mmcif_path = os.path.join(data_dir, f"{file_name}.cif")
        if not os.path.exists(mmcif_path):
            logger.warning(f"Could not find {mmcif_path}, skip!")
            continue
        else:
            valid_file_name_list.append(file_name)
        pdb_path = os.path.join(data_dir, "converted_pdbs", f"{file_name}.pdb")
        os.makedirs(os.path.dirname(pdb_path), exist_ok=True)
        new_cond_chains, new_binder_chains = convert_cif_to_pdb(
            cif_path=mmcif_path,
            out_pdb_path=pdb_path,
            binder_chains=binder_chains,
        )
        if n_cond_chains is None:
            n_cond_chains = list(new_cond_chains)
        else:
            assert set(n_cond_chains) == set(new_cond_chains)
        if n_binder_chains is None:
            n_binder_chains = list(new_binder_chains)
        else:
            assert set(n_binder_chains) == set(new_binder_chains)

    logger.info(
        f"Found {len(valid_file_name_list)} valid files, cond chains: {n_cond_chains}, binder chains: {n_binder_chains}"
    )
    input_data = {
        "task": "binder",
        "name": os.path.basename(data_dir),
        "pdb_dir": os.path.join(data_dir, "converted_pdbs"),
        "pdb_names": valid_file_name_list,
        "cond_chains": n_cond_chains,
        "binder_chains": n_binder_chains,
        "out_dir": dump_dir,
    }
    return input_data


def prepare_tasks_from_pdb(
    data_dir: str,
    file_name_list: list,
    binder_chains: list[str],
    cond_chains: list[str],
    dump_dir: str,
):
    n_cond_chains = None
    valid_file_name_list = []
    for file_name in file_name_list:
        pdb_path = os.path.join(data_dir, f"{file_name}.pdb")
        if not os.path.exists(pdb_path):
            logger.warning(f"Could not find {pdb_path}, skip!")
            continue
        else:
            valid_file_name_list.append(file_name)

        if cond_chains == [""]:
            chains = get_chains_from_pdb(pdb_path)
            new_cond_chains = chains - set(binder_chains)
        else:
            new_cond_chains = cond_chains
        if n_cond_chains is None:
            n_cond_chains = list(new_cond_chains)
        else:
            assert set(n_cond_chains) == set(new_cond_chains)

    return {
        "task": "binder",
        "pdb_dir": data_dir,
        "name": os.path.basename(data_dir),
        "pdb_names": valid_file_name_list,
        "cond_chains": n_cond_chains,
        "binder_chains": binder_chains,
        "out_dir": dump_dir,
    }


def prepare_tasks_from_json(json_path):
    """
    Prepare tasks from a JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        list: List of tasks.
    """
    with open(json_path, "r") as f:
        tasks = json.load(f)
    if isinstance(tasks, dict):
        tasks = [tasks]
    return tasks


def main():
    # Configs
    config_dict = {
        "file_name_list": "",
        "json_path": "",
        "data_dir": "",
        "dump_dir": RequiredValue(str),
        "cond_chains": ListValue([""]),
        "binder_chains": ListValue([""]),  # required
        "is_mmcif": False,
        "orig_seqs_json": "",
        "seed": 2025,
        **eval_configs,
    }
    configs = parse_configs(config_dict, arg_str=parse_sys_args())

    # Prepare tasks
    if configs.json_path:
        logger.info("Prepare tasks from json file.")
        input_data_list = prepare_tasks_from_json(configs.json_path)
        for input_data in input_data_list:
            run_task(input_data, configs)

    else:
        if not configs.file_name_list:
            if configs.is_mmcif:
                fn_list = find_files_with_ext(configs.data_dir, "cif")
            else:
                fn_list = find_files_with_ext(configs.data_dir, "pdb")
        else:
            fn_list = get_file_name_list(configs.file_name_list)
        if configs.is_mmcif:
            logger.info("Prepare tasks from mmCIF files.")
            input_data = prepare_tasks_from_mmcif(
                data_dir=configs.data_dir,
                file_name_list=fn_list,
                binder_chains=configs.binder_chains,
                cond_chains=configs.cond_chains,
                dump_dir=configs.dump_dir,
            )
        else:
            logger.info("Prepare tasks from PDB files.")
            input_data = prepare_tasks_from_pdb(
                data_dir=configs.data_dir,
                file_name_list=fn_list,
                binder_chains=configs.binder_chains,
                cond_chains=configs.cond_chains,
                dump_dir=configs.dump_dir,
            )
        if len(configs.orig_seqs_json) > 0:
            input_data["orig_seqs_json"] = configs.orig_seqs_json
        # Run task
        run_task(input_data, configs, seed=configs.seed)


if __name__ == "__main__":
    main()
