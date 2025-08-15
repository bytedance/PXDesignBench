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

import ast
import json
import logging
import os
import random
import re
import subprocess
import sys
import urllib.request
from copy import deepcopy
from datetime import datetime
from os.path import exists as opexists
from typing import Any, Dict, FrozenSet, Iterable, List, Set, Tuple, Union

import torch
from ml_collections.config_dict import ConfigDict
from protenix.config import parse_configs
from protenix.web_service.dependency_url import URL

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs

logger = logging.getLogger(__name__)


def download_infercence_cache(configs: Any) -> None:
    def progress_callback(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        bar_length = 30
        filled_length = int(bar_length * percent // 100)
        bar = "=" * filled_length + "-" * (bar_length - filled_length)

        status = f"\r[{bar}] {percent:.1f}%"
        print(status, end="", flush=True)

        if downloaded >= total_size:
            print()

    def download_from_url(tos_url, checkpoint_path, check_weight=True):
        urllib.request.urlretrieve(
            tos_url, checkpoint_path, reporthook=progress_callback
        )
        if check_weight:
            try:
                ckpt = torch.load(checkpoint_path)
                del ckpt
            except:
                os.remove(checkpoint_path)
                raise RuntimeError(
                    "Download model checkpoint failed, please download by yourself with "
                    f"wget {tos_url} -O {checkpoint_path}"
                )

    for cache_name in (
        "ccd_components_file",
        "ccd_components_rdkit_mol_file",
        "pdb_cluster_file",
    ):
        cur_cache_fpath = configs["data"][cache_name]
        if not opexists(cur_cache_fpath):
            os.makedirs(os.path.dirname(cur_cache_fpath), exist_ok=True)
            tos_url = URL[cache_name]
            assert os.path.basename(tos_url) == os.path.basename(cur_cache_fpath), (
                f"{cache_name} file name is incorrect, `{tos_url}` and "
                f"`{cur_cache_fpath}`. Please check and try again."
            )
            logger.info(
                f"Downloading data cache from\n {tos_url}... to {cur_cache_fpath}"
            )
            download_from_url(tos_url, cur_cache_fpath, check_weight=False)

    checkpoint_path = f"{configs.load_checkpoint_dir}/{configs.model_name}.pt"
    checkpoint_dir = configs.load_checkpoint_dir

    if not opexists(checkpoint_path):
        os.makedirs(checkpoint_dir, exist_ok=True)
        tos_url = URL[configs.model_name]
        logger.info(
            f"Downloading model checkpoint from\n {tos_url}... to {checkpoint_path}"
        )
        download_from_url(tos_url, checkpoint_path)

    if "esm" in configs.model_name:  # currently esm only support 3b model
        esm_3b_ckpt_path = f"{checkpoint_dir}/esm2_t36_3B_UR50D.pt"
        if not opexists(esm_3b_ckpt_path):
            tos_url = URL["esm2_t36_3B_UR50D"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}... to {esm_3b_ckpt_path}"
            )
            download_from_url(tos_url, esm_3b_ckpt_path)
        esm_3b_ckpt_path2 = f"{checkpoint_dir}/esm2_t36_3B_UR50D-contact-regression.pt"
        if not opexists(esm_3b_ckpt_path2):
            tos_url = URL["esm2_t36_3B_UR50D-contact-regression"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}... to {esm_3b_ckpt_path2}"
            )
            download_from_url(tos_url, esm_3b_ckpt_path2)
    if "ism" in configs.model_name:
        esm_3b_ism_ckpt_path = f"{checkpoint_dir}/esm2_t36_3B_UR50D_ism.pt"

        if not opexists(esm_3b_ism_ckpt_path):
            tos_url = URL["esm2_t36_3B_UR50D_ism"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}... to {esm_3b_ism_ckpt_path}"
            )
            download_from_url(tos_url, esm_3b_ism_ckpt_path)

        esm_3b_ism_ckpt_path2 = f"{checkpoint_dir}/esm2_t36_3B_UR50D_ism-contact-regression.pt"  # the same as esm_3b_ckpt_path2
        if not opexists(esm_3b_ism_ckpt_path2):
            tos_url = URL["esm2_t36_3B_UR50D_ism-contact-regression"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}... to {esm_3b_ism_ckpt_path2}"
            )
            download_from_url(tos_url, esm_3b_ism_ckpt_path2)


def get_configs(model_name):
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        fill_required_with_null=True,
    )
    model_specfics_configs = ConfigDict(model_configs[model_name])
    configs.model_name
    # update model specific configs
    configs.update(model_specfics_configs)
    return configs


def _get_entity_key(seq_entry: dict) -> str:
    """
    A seq_entry must be a one-key dict (e.g., {"proteinChain": {...}}).
    Return that single key.
    """
    if not isinstance(seq_entry, dict) or len(seq_entry) != 1:
        raise ValueError("seq_entry must be a dict with exactly one top-level key.")
    return next(iter(seq_entry.keys()))


def _get_entity_dict(seq_entry: dict) -> dict:
    """Return the inner entity dict (e.g., seq_entry['proteinChain'])."""
    return seq_entry[_get_entity_key(seq_entry)]


def _split_seq_entry_by_asym(
    seq_entry: dict, subset_asym_groups: Iterable[Iterable[str]]
) -> List[dict]:
    """
    Split a seq_entry into multiple entries based on provided asym subsets.
    - Each subset becomes a new entry with label_asym_id = that subset and count = len(subset).
    - If the union of subsets doesn't cover all original asym IDs, add a remainder entry.
    """
    entity_key = _get_entity_key(seq_entry)
    entity = _get_entity_dict(seq_entry)

    if "label_asym_id" not in entity or not isinstance(entity["label_asym_id"], list):
        raise ValueError("Entity must contain a 'label_asym_id' list.")

    orig_asym_list = entity["label_asym_id"]
    if len(set(orig_asym_list)) != len(orig_asym_list):
        raise ValueError("Original 'label_asym_id' contains duplicates.")
    orig_asym_set: Set[str] = set(orig_asym_list)

    # Normalize and validate subsets
    normalized_groups: List[Set[str]] = []
    seen: Set[str] = set()
    for group in subset_asym_groups:
        gset = set(group)
        if not gset:
            continue
        if not gset.issubset(orig_asym_set):
            raise ValueError(
                f"Subset {gset} is not a subset of original asym IDs {orig_asym_set}."
            )
        if seen & gset:
            raise ValueError(f"Subset {gset} overlaps with previous subsets {seen}.")
        seen |= gset
        normalized_groups.append(gset)

    results: List[dict] = []
    # Create entries for each provided subset
    for gset in normalized_groups:
        new_entry = deepcopy(seq_entry)
        new_entity = new_entry[entity_key]
        new_entity["label_asym_id"] = sorted(gset)
        new_entity["count"] = len(gset)
        results.append(new_entry)

    # Add remainder if any asym IDs are uncovered
    remaining = orig_asym_set - seen
    if remaining:
        remainder_entry = deepcopy(seq_entry)
        remainder_entity = remainder_entry[entity_key]
        remainder_entity["label_asym_id"] = sorted(remaining)
        remainder_entity["count"] = len(remaining)
        results.append(remainder_entry)

    return results


def _build_asym_index(sequences: List[dict], trim=False) -> Dict[FrozenSet[str], dict]:
    """
    Build {frozenset(asym_ids): seq_entry} index.
    Enforces:
      - Each seq_entry has a 'label_asym_id' list.
      - No duplicates inside a seq_entry.
      - No overlap across entries (disjoint partition of asym IDs).
    """
    idx: Dict[FrozenSet[str], dict] = {}
    used: Set[str] = set()

    for seq_entry in sequences:
        entity = _get_entity_dict(seq_entry)
        if "label_asym_id" not in entity or not isinstance(
            entity["label_asym_id"], list
        ):
            raise ValueError(
                "Each seq_entry entity must contain a 'label_asym_id' list."
            )

        asyms: List[str] = entity["label_asym_id"]
        if trim:
            asyms = [a[0] for a in asyms]
        if len(set(asyms)) != len(asyms):
            raise ValueError(f"Entry has duplicate asym IDs: {asyms}")

        aset = set(asyms)
        overlap = used & aset
        if overlap:
            raise ValueError(f"Overlapping asym IDs across entries: {sorted(overlap)}")
        used |= aset

        idx[frozenset(aset)] = seq_entry

    return idx


def patch_with_orig_seqs(sample_list: List[dict], orig_seqs: list) -> List[dict]:
    """
    For each item in sample_list:
      1) Build an index of current sequences grouped by their asym sets.
      2) Build an index of the original sequences grouped by their asym sets
      3) For each original asym set, find a current asym set that contains it (superset). If none, raise.
      4) For each current asym set that has matches:
           - Split the current entry into those subsets (+ remainder if needed).
           - For each split piece, if it exactly matches an original subset, copy FIELDS_TO_COPY.
         Else, keep the current entry as-is.

    Returns a deep-copied transformed list.
    """
    FIELDS_TO_COPY = ["sequence", "use_msa", "msa", "crop"]
    out = deepcopy(sample_list)

    # Build original index once (applies to each item)
    orig_idx = _build_asym_index(orig_seqs, trim=True)
    orig_sets: List[FrozenSet[str]] = list(orig_idx.keys())

    for i, item in enumerate(out):
        if "sequences" not in item or not isinstance(item["sequences"], list):
            raise ValueError(f"Item #{i} missing a valid 'sequences' list.")

        cur_idx = _build_asym_index(item["sequences"])

        # Map: current_asym_set -> list of original_asym_sets that are subsets of that current set
        matched: Dict[FrozenSet[str], List[FrozenSet[str]]] = {}
        for oset in orig_sets:
            container = None
            for cset in cur_idx.keys():
                if oset.issubset(cset):
                    container = cset
                    break
            if container is None:
                raise ValueError(
                    f"Item #{i}: missing container for original subset {sorted(oset)}."
                )
            matched.setdefault(container, []).append(oset)

        new_sequences: List[dict] = []

        for cset, cur_entry in cur_idx.items():
            if cset in matched:
                # Split current entry by matched subsets (may produce a remainder)
                subset_groups = [list(s) for s in matched[cset]]
                split_entries = _split_seq_entry_by_asym(cur_entry, subset_groups)

                for split_entry in split_entries:
                    entity_key = _get_entity_key(split_entry)
                    entity = split_entry[entity_key]
                    saset = frozenset(entity["label_asym_id"])

                    # Copy fields only if the split matches an original subset
                    if saset in orig_idx:
                        orig_entity = _get_entity_dict(orig_idx[saset])
                        for k in FIELDS_TO_COPY:
                            if k in orig_entity:
                                entity[k] = deepcopy(orig_entity[k])

                    new_sequences.append(split_entry)
            else:
                # No matching subsets -> keep as-is
                new_sequences.append(cur_entry)

        item["sequences"] = new_sequences

    return out


def _random_suffix(length=6):
    """Generate a short random hex string."""
    return "".join(random.choices("0123456789abcdef", k=length))


def run_protenix_msa(cmd: list[str]) -> Dict[str, str]:
    """
    Run the command, print its stdout in real-time,
    then parse the last {...} in stdout as a Python dict.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    buf_lines = []

    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)  # Real-time display
        buf_lines.append(line)

    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    stdout_text = "".join(buf_lines)

    # Extract last {...}
    brace_blocks = re.findall(r"\{.*\}", stdout_text, flags=re.DOTALL)
    if not brace_blocks:
        raise RuntimeError("No dictionary-like {...} found in output.")

    last_block = brace_blocks[-1]
    try:
        return ast.literal_eval(last_block)
    except Exception as e:
        raise RuntimeError(f"Failed to parse last dict from output: {e}")


def populate_msa_with_cache(
    data: List[Dict],
    *,
    cache_file: str = "./msa_cache/cache.json",
    out_dir: str = "./msa_cache",
) -> List[Dict]:
    """
    Same as before, but fasta_path is placed in a short human-readable subdir:
    YYYYMMDD_<random_hex>/input.fasta
    """

    def _iter_entities(items: List[dict]):
        for i, item in enumerate(items):
            for j, seq_entry in enumerate(item.get("sequences", [])):
                if not isinstance(seq_entry, dict) or len(seq_entry) != 1:
                    continue
                entity = next(iter(seq_entry.values()))
                yield i, j, entity

    def _sanitize_sequence(seq: str) -> str:
        return "".join(seq.split()).upper()

    def _needs_msa(entity: dict) -> bool:
        use_msa = entity.get("use_msa", True)
        if not use_msa:
            return False
        msa = entity.get("msa", {})
        precomp = msa.get("precomputed_msa_dir") if isinstance(msa, dict) else None
        return not precomp

    # Ensure base dirs exist
    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Load cache
    cache: Dict[str, str] = {}
    if os.path.isfile(cache_file):
        try:
            with open(cache_file, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    cache = {
                        _sanitize_sequence(k): v
                        for k, v in loaded.items()
                        if isinstance(k, str)
                    }
        except Exception:
            cache = {}

    # Collect needed sequences
    pending: Set[str] = set()
    wanted_pairs: List[Tuple[int, int, str]] = []
    for i, j, entity in _iter_entities(data):
        if not _needs_msa(entity):
            continue
        seq = entity.get("sequence")
        if not isinstance(seq, str):
            continue
        sseq = _sanitize_sequence(seq)
        if not sseq:
            continue
        wanted_pairs.append((i, j, sseq))
        if sseq not in cache:
            pending.add(sseq)

    # If pending, run MSA
    if pending:
        # Create short readable random subdir
        date_str = datetime.now().strftime("%Y%m%d")
        random_dir = os.path.join(out_dir, f"{date_str}_{_random_suffix()}")
        os.makedirs(random_dir, exist_ok=True)
        fasta_path = os.path.join(random_dir, "input.fasta")

        # Write FASTA
        with open(fasta_path, "w") as f:
            for idx, sseq in enumerate(sorted(pending)):
                f.write(f">seq_{idx+1}\n")
                f.write(sseq + "\n")

        # Run protenix msa
        print(f"Searching MSA with input fasta {fasta_path} and out dir {random_dir}")
        cmd = ["protenix", "msa", "--input", fasta_path, "--out_dir", random_dir]
        returned = run_protenix_msa(cmd)

        # Merge into cache
        for seq_key, msa_path in returned.items():
            sseq = _sanitize_sequence(seq_key)
            if sseq in pending:
                cache[sseq] = msa_path

        # Save cache
        tmp_path = cache_file + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp_path, cache_file)

    # Produce updated data
    new_data = deepcopy(data)
    for i, j, entity in _iter_entities(new_data):
        if not _needs_msa(entity):
            continue
        sseq = _sanitize_sequence(entity["sequence"])
        if sseq in cache:
            if "msa" not in entity or not isinstance(entity["msa"], dict):
                entity["msa"] = {}
            entity["msa"]["precomputed_msa_dir"] = cache[sseq]
            entity["msa"]["pairing_db"] = "uniref100"

    return new_data
