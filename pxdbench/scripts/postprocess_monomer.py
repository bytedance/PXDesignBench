import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np

from pxdbench.metrics.diversity import compute_diversity
from pxdbench.utils import str2bool

FOLDSEEK_BIN = "your_foldseek_dir/bin/foldseek"
FOLDSEEK_DB = "your_foldseek_dir/foldseek_db/pdb/pdb"


def compute_fs_diversity(input_dir: Path, num_threads=32):
    num_pdbs = sum(1 for f in input_dir.glob("*.pdb") if f.is_file())
    cluster_out = input_dir / "fs_diversity" / "res"
    tmp_dir = input_dir / "fs_diversity_tmp"
    cluster_out.parent.mkdir(parents=True, exist_ok=True)
    cluster_tsv = cluster_out.with_name(cluster_out.stem + "_cluster.tsv")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if not cluster_tsv.exists():
        subprocess.run(
            [
                FOLDSEEK_BIN,
                "easy-cluster",
                str(input_dir),
                str(cluster_out),
                str(tmp_dir),
                "--alignment-type",
                "1",
                "--cov-mode",
                "0",
                "--min-seq-id",
                "0",
                "--tmscore-threshold",
                "0.5",
                "--threads",
                f"{num_threads}",
            ],
            check=True,
        )
    # Count clusters and samples
    num_clusters = 0
    seen_clusters = set()
    if cluster_tsv.exists():
        with open(cluster_tsv) as f:
            for line in f:
                cluster_name, member = line.strip().split("\t")
                if cluster_name not in seen_clusters:
                    seen_clusters.add(cluster_name)
                    num_clusters += 1
    diversity_cluster = num_clusters / max(num_pdbs, 1)
    shutil.rmtree(tmp_dir)
    return diversity_cluster


def compute_fs_novelty(input_dir: Path, use_gpu=True, num_threads=32):
    tmp_dir = input_dir / "fs_novelty_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    novelty_out = input_dir / "fs_novelty" / "novelty.tsv"
    novelty_out.parent.mkdir(parents=True, exist_ok=True)

    if not novelty_out.exists():
        cmd = [
            FOLDSEEK_BIN,
            "easy-search",
            str(input_dir),
            FOLDSEEK_DB,
            str(novelty_out),
            str(tmp_dir),
            "--alignment-type",
            "1",
            "--exhaustive-search",
            "--tmscore-threshold",
            "0.0",
            "--max-seqs",
            "10000000000",
            "--format-output",
            "query,target,alntmscore,lddt",
            "--threads",
            f"{num_threads}",
        ]
        if use_gpu:
            cmd.extend(["--gpu", "1", "--prefilter-mode", "1"])
        subprocess.run(
            cmd,
            check=True,
        )

    max_scores = {}
    if novelty_out.exists():
        with open(novelty_out) as f:
            for line in f:
                query, _, tmscore, _ = line.strip().split("\t")
                tmscore = float(tmscore)
                if query not in max_scores or tmscore > max_scores[query]:
                    max_scores[query] = tmscore
    novelty_score = np.mean(list(max_scores.values()))
    shutil.rmtree(tmp_dir)
    return novelty_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_novelty", action="store_true", default=False)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--num_threads", type=int, default=32)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input dir does not exist! {input_dir}")
        return
    if args.output_dir is None:
        output_dir = input_dir / "postprocess"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / "diversity_and_novelty.csv"
    fieldnames = ["num_samples", "diversity_tm", "diversity_cluster", "novelty"]

    # scan PDBs
    pdb_paths = sorted(input_dir.glob("*.pdb"))
    pdb_paths = [p for p in pdb_paths if p.is_file()]
    if len(pdb_paths) == 0:
        print(f"No PDB files found in {input_dir}")
        # wrtie an empty file
        with open(output_csv_path, "w") as f:
            f.write(",".join(fieldnames) + "\n")
            f.write("0,,,\n")
        return

    # Diversity (TM)
    try:
        diversity_tm = compute_diversity(pdb_paths) if len(pdb_paths) >= 2 else np.nan
    except Exception as e:
        print(f"compute_diversity failed: {e}")
        diversity_tm = np.nan

    # Diversity (Cluster, Foldseek)
    try:
        diversity_cluster = compute_fs_diversity(
            input_dir, num_threads=args.num_threads
        )
    except Exception as e:
        print(f"compute_fs_diversity failed: {e}")
        diversity_cluster = np.nan

    novelty = np.nan
    if args.eval_novelty:
        if not Path(FOLDSEEK_BIN).exists():
            print(f"Foldseek binary not found: {FOLDSEEK_BIN}, skip novelty.")
        elif (
            not Path(FOLDSEEK_DB + ".dbtype").exists()
            and not Path(FOLDSEEK_DB).exists()
        ):
            print(f"Foldseek DB not found: {FOLDSEEK_DB}, skip novelty.")
        else:
            try:
                novelty = compute_fs_novelty(
                    input_dir, args.use_gpu, num_threads=args.num_threads
                )
            except Exception as e:
                print(f"compute_fs_novelty failed: {e}")
                novelty = np.nan

    with open(output_csv_path, "w") as f:
        f.write(",".join(fieldnames) + "\n")
        f.write(f"{len(pdb_paths)},{diversity_tm},{diversity_cluster},{novelty}\n")
    print(f"Wrote results to {output_csv_path}")


if __name__ == "__main__":
    main()
