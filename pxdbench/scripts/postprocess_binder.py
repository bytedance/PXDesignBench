import argparse
import multiprocessing as mp
import os
import subprocess
import warnings
from glob import glob
from itertools import combinations

import numpy as np
from biotite.structure.io import load_structure
from biotite.structure.io.pdb import PDBFile

from pxdbench.globals import TMALIGN_PATH
from pxdbench.utils import convert_cifs_to_pdbs, str2bool

warnings.filterwarnings("ignore", module="biotite")


def run_tmalign(pdb1, pdb2, tmalign_path="TMalign"):
    """
    Run TM-align on two PDB files and return the output.

    Args:
        pdb1 (str): Path to the first PDB file
        pdb2 (str): Path to the second PDB file
        tmalign_path (str): Path to the TM-align executable

    Returns:
        str: The output from TM-align
    """
    cmd = [tmalign_path, pdb1, pdb2]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running TMalign: {e}")
        return None


def extract_tmscore(tmalign_output):
    """
    Extract the TM-score from the TM-align output.

    Args:
        tmalign_output (str): The output from TM-align

    Returns:
        float: The TM-score (average of the two normalized scores), or None if not found
    """
    if tmalign_output is None:
        return None

    # TM-align returns two TM-scores (normalized by different lengths)
    tm_scores = []
    for line in tmalign_output.split("\n"):
        if "TM-score=" in line:
            try:
                tm_score = float(line.split("=")[1].split("(")[0].strip())
                tm_scores.append(tm_score)
            except (IndexError, ValueError):
                continue

    if tm_scores:
        return sum(tm_scores) / len(tm_scores)
    return None


def cluster_worker(args):
    pdb1, pdb2, i, j, tmalign_path = args
    tmalign_output = run_tmalign(pdb1, pdb2, tmalign_path)
    tm_score = extract_tmscore(tmalign_output)
    return (i, j, tm_score)


def calculate_pairwise_tmscores(pdb_files, tmalign_path):
    """
    Calculate pairwise TM-scores for a list of PDB files, in parallel.

    Args:
        pdb_files (list): List of paths to PDB files
        tmalign_path (str): Path to the TM-align executable

    Returns:
        numpy.ndarray: Matrix of pairwise TM-scores
    """
    n = len(pdb_files)
    tm_matrix = np.zeros((n, n))

    # Set diagonal to 1.0
    np.fill_diagonal(tm_matrix, 1.0)

    # Create list of pairs (i, j)
    pairs = list(combinations(range(n), 2))

    # Prepare arguments for multiprocessing
    args_list = [(pdb_files[i], pdb_files[j], i, j, tmalign_path) for i, j in pairs]
    num_workers = min(40, mp.cpu_count())

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(cluster_worker, args_list)

    for i, j, tm_score in results:
        if tm_score is not None:
            tm_matrix[i, j] = tm_score
            tm_matrix[j, i] = tm_score

    return tm_matrix


def greedy_clustering(pdb_files, tm_matrix, threshold):
    """
    Perform greedy clustering based on TM-scores.

    This implementation selects the structure with the most
    unassigned neighbors above the threshold as the next cluster center.

    Args:
        pdb_files (list): List of paths to PDB files
        tm_matrix (numpy.ndarray): Matrix of pairwise TM-scores
        threshold (float): TM-score threshold for clustering

    Returns:
        list: List of clusters, where each cluster is a list of PDB file indices
    """
    n = len(pdb_files)
    assigned = [False] * n
    clusters = []

    print(f"Performing greedy clustering with TM-score threshold {threshold}...")

    while not all(assigned):
        # Find unassigned structure with most unassigned neighbors
        max_neighbors = -1
        center_idx = -1

        for i in range(n):
            if assigned[i]:
                continue

            # Count unassigned neighbors (including self)
            count = sum(
                1 for j in range(n) if not assigned[j] and tm_matrix[i, j] >= threshold
            )

            if count > max_neighbors:
                max_neighbors = count
                center_idx = i

        if center_idx == -1:
            break  # No unassigned structures left

        # Create new cluster
        current_cluster = [center_idx]
        assigned[center_idx] = True

        # Add all similar unassigned structures to the cluster
        for j in range(n):
            if not assigned[j] and tm_matrix[center_idx, j] >= threshold:
                current_cluster.append(j)
                assigned[j] = True

        clusters.append(current_cluster)
        print(f"Created cluster {len(clusters)} with {len(current_cluster)} structures")

    return clusters


def save_tm_matrix(tm_matrix, pdb_files, output_file):
    """
    Save the TM-score matrix to a file.

    Args:
        tm_matrix (numpy.ndarray): Matrix of pairwise TM-scores
        pdb_files (list): List of paths to PDB files
        output_file (str): Path to the output file
    """
    with open(output_file, "w") as f:
        # Write header
        f.write("# TM-score matrix\n")
        f.write("# Format: <pdb_i> <pdb_j> <tm_score>\n\n")

        n = len(pdb_files)
        for i in range(n):
            for j in range(i, n):  # Only upper triangle including diagonal
                pdb_i = os.path.basename(pdb_files[i])
                pdb_j = os.path.basename(pdb_files[j])
                tm_score = tm_matrix[i, j]
                f.write(f"{pdb_i} {pdb_j} {tm_score:.4f}\n")


def extract_chain_from_pdb(
    pdb_dir: str,
    pdb_files: list[str],
    chain_id: str,
):
    new_file_list = []
    folder_path = f"{pdb_dir}/tmp"
    os.makedirs(folder_path, exist_ok=True)
    for input_file_path in pdb_files:
        name = input_file_path.split("/")[-1].split(".")[0]
        pdb_file_path = f"{pdb_dir}/tmp/{name}.pdb"

        # we only consider input pdb file
        structure = load_structure(input_file_path)
        # make sure chain id is in structure
        chain_structure = structure[structure.chain_id == chain_id]
        if len(chain_structure) == 0:
            raise ValueError(f"chain {chain_id} not in {input_file_path}")

        pdb_file = PDBFile()
        pdb_file.set_structure(chain_structure)
        pdb_file.write(pdb_file_path)
        new_file_list.append(pdb_file_path)
    print(f"finish extract chain")
    return folder_path, new_file_list


def main():
    parser = argparse.ArgumentParser(
        description="Perform pairwise TM-align and greedy clustering of PDB files"
    )
    parser.add_argument(
        "--input_dir", required=True, help="Directory containing PDB/CIF files"
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="TM-score threshold for clustering (default: 0.5)",
    )
    parser.add_argument(
        "--clusters_output",
        default="clusters.txt",
        help="Output file for clustering results (default: clusters.txt)",
    )
    parser.add_argument(
        "--matrix_output",
        default="tm_matrix.txt",
        help="Output file for TM-score matrix (default: tm_matrix.txt)",
    )
    parser.add_argument(
        "--tmalign_path",
        default=TMALIGN_PATH,
        help="Path to the TM-align executable (default: TMalign in PATH)",
    )
    parser.add_argument(
        "--is_mmcif", default=False, type=str2bool, help="input file type, mmcif or pdb"
    )
    parser.add_argument("--binder_chain", default=None, help="only calculate one chain")

    args = parser.parse_args()

    # Get all PDB files in the directory
    if args.is_mmcif:
        pdb_dir, pdb_names, _, _ = convert_cifs_to_pdbs(
            args.input_dir,
            out_pdb_dir=os.path.join(args.input_dir, "converted_pdbs"),
        )
        pdb_files = sorted([os.path.join(pdb_dir, fn + ".pdb") for fn in pdb_names])
    else:
        pdb_dir = args.input_dir
        pdb_files = sorted(glob(os.path.join(args.input_dir, "*.pdb")))

    if not pdb_files:
        print(f"No PDB files found in {args.input_dir}")
        return

    if args.binder_chain is not None:
        if len(args.binder_chain) > 1:
            args.binder_chain = args.binder_chain[0]
            print(
                f"Use the chain ID in the PDB file -- trim it to one char: {args.binder_chain}"
            )

        pdb_dir, pdb_files = extract_chain_from_pdb(
            pdb_dir, pdb_files, args.binder_chain
        )

    print(f"Found {len(pdb_files)} PDB files")

    if args.output_dir is None:
        output_dir = os.path.join(args.input_dir, "postprocess")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Calculate pairwise TM-scores
    tm_matrix = calculate_pairwise_tmscores(pdb_files, args.tmalign_path)

    # Save the TM-score matrix
    save_tm_matrix(tm_matrix, pdb_files, os.path.join(output_dir, args.matrix_output))
    print(f"TM-score matrix saved to {os.path.join(output_dir, args.matrix_output)}")

    # Perform greedy clustering
    clusters = greedy_clustering(pdb_files, tm_matrix, args.threshold)

    # Write clustering results
    with open(os.path.join(output_dir, args.clusters_output), "w") as f:
        f.write(f"# Clustering with TM-score threshold: {args.threshold}\n")
        f.write(f"# Number of clusters: {len(clusters)}\n\n")

        for i, cluster in enumerate(clusters):
            f.write(f"Cluster {i+1} (size: {len(cluster)}):\n")
            # Write the representative (center) first
            f.write(f"  {os.path.basename(pdb_files[cluster[0]])} (center)\n")
            # Write the rest of the cluster members
            for idx in cluster[1:]:
                f.write(f"  {os.path.basename(pdb_files[idx])}\n")
            f.write("\n")

    print(f"Clustering completed. Found {len(clusters)} clusters.")
    print(f"Results written to {os.path.join(output_dir, args.clusters_output)}")


if __name__ == "__main__":
    main()
