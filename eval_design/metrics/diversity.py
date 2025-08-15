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

import multiprocessing as mp
from itertools import combinations

from eval_design.metrics import tmalign


def compute_pair_tm_score(args):
    file1, file2, chain_id = args
    tm_score = tmalign.get_tm_score(file1, file2, chain_id, chain_id)
    return tm_score


def compute_diversity(pdb_files, chain_id=None):
    """
    Calculate the average TM-score between all pairs of PDB files to measure diversity.

    Args:
        pdb_files (list): A list of file paths to PDB files.
        chain_id (str, optional): The chain ID to use for TM-score calculation. Defaults to None.

    Returns:
        float or None: The average TM-score between all valid pairs of PDB files.
                       Returns None if no valid pairs are found.
    """
    num_workers = min(32, mp.cpu_count())
    pairs = list(combinations(pdb_files, 2))

    with mp.Pool(processes=num_workers) as pool:
        args_list = [(file1, file2, chain_id) for file1, file2 in pairs]
        tm_scores = pool.map(compute_pair_tm_score, args_list)

    valid_scores = [score for score in tm_scores if score is not None]
    count = len(valid_scores)
    sum_scores = sum(valid_scores)

    if count > 0:
        average = sum_scores / count
        print(f"avg pair TMscore: {average:.4f}")
        print(f"total nums pair: {count}")
    else:
        average = None
        print("No valid TM-score pairs found.")
    return average
