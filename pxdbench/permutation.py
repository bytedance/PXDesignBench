import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from Bio.PDB import PDBIO
from Bio.PDB import MMCIFParser as BioMMCIFParser
from Bio.PDB import PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model

try:
    from Bio.PDB.Polypeptide import three_to_one  # works on some older versions
except ImportError:
    from Bio.SeqUtils import seq1 as three_to_one  # fallback on newer versions

from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

# Your project (Biotite-backed) I/O for mmCIF
from protenix.data.parser import MMCIFParser  # loads AtomArray
from protenix.data.utils import CIFWriter
from scipy.optimize import linear_sum_assignment

from pxdbench.metrics.Kalign import kabsch_algorithm

# ------------------------- basic utils -------------------------


def is_cif(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".cif", ".mmcif"}


def is_pdb(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".pdb", ".ent"}


def is_polymer_res(res) -> bool:
    """Polymer residues only (ATOM, hetflag == ' ')."""
    hetflag, _, _ = res.id
    return hetflag == " "


def polymer_chains(model):
    """List polymer chains (at least one polymer residue)."""
    return [ch for ch in model if any(is_polymer_res(r) for r in ch)]


def chain_seq_1letter(chain) -> str:
    """One-letter sequence for a chain (polymer residues only). Non-standard -> 'X'."""
    seq = []
    for res in chain:
        if not is_polymer_res(res):
            continue
        rn = res.get_resname().strip()
        try:
            aa = three_to_one(rn)
        except KeyError:
            aa = "X"
        seq.append(aa)
    return "".join(seq)


def extract_ca_coords(chain) -> np.ndarray:
    """(N,3) CA coordinates for polymer residues; (0,3) if empty."""
    coords = []
    for res in chain:
        if not is_polymer_res(res):
            continue
        if "CA" in res:
            coords.append(res["CA"].coord)
    if not coords:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(coords).astype(float)


# ------------------------- loading models -------------------------


def load_biopython_model(path: str):
    """
    Load a structure with Biopython and return (structure, model).
    Supports PDB and mmCIF.
    """
    if is_cif(path):
        parser = BioMMCIFParser(QUIET=True)
        struct = parser.get_structure("mmcif", path)
    elif is_pdb(path):
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("pdb", path)
    else:
        raise ValueError(f"Unsupported format for: {path}")
    model = next(struct.get_models())
    return struct, model


# ------------------------- rigid transform & RMSD -------------------------


def fit_rotran_from_pair(G: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rigid transform (rotation, translation) aligning G -> R using CA pairs.
    Returns rotation (3x3) and translation (3,). Requires >=3 points (otherwise skip).
    """
    n = min(len(G), len(R))
    if n < 3:
        raise ValueError(
            "Not enough CA to compute a stable rigid transform (need >=3)."
        )
    R, C_P, C_Q = kabsch_algorithm(R[:n], G[:n])
    return R, C_P, C_Q


def apply_rotran(
    coords: np.ndarray, R: np.ndarray, C_P: np.ndarray, C_Q: np.ndarray
) -> np.ndarray:
    """Apply rot, tran to an (N,3) array"""
    if coords.size == 0:
        return coords
    return np.dot(coords - C_Q, R) + C_P


def complex_rmsd_under_transform(
    gen_coords_list: List[np.ndarray],
    ref_coords_list: List[np.ndarray],
    mapping: Dict[int, int],
    rot: np.ndarray,
    tr_p: np.ndarray,
    tr_q: np.ndarray,
) -> float:
    """Complex CA RMSD after applying (rot, tran) and pairing by mapping (gen_idx -> ref_idx)."""
    sum_sq, n_pts = 0.0, 0
    for gi, rj in mapping.items():
        G = apply_rotran(gen_coords_list[gi], rot, tr_p, tr_q)
        R = ref_coords_list[rj]
        n = min(len(G), len(R))
        if n == 0:
            continue
        diff = G[:n] - R[:n]
        sum_sq += float((diff * diff).sum())
        n_pts += n
    if n_pts == 0:
        return float("inf")
    return np.sqrt(sum_sq / n_pts)


# ------------------------- anchored mapping (reduced enumeration) -------------------------


def anchored_optimal_mapping_grouped(
    ref_model, gen_model
) -> Tuple[Dict[str, str], float]:
    """
    Reduced-anchor search:
      - Group chains by identical sequence.
      - For each sequence group s:
          pick ONE generated representative (first index) as anchor,
          try all reference chains in that group,
          compute (rot, tran) from the anchor pair,
          build cost matrix under fixed transform (only same-sequence pairs allowed),
          solve Hungarian, compute complex RMSD,
          keep the overall best mapping across all groups/anchors.
    Returns:
      (rename_map {gen_chain_id -> ref_chain_id}, best_complex_RMSD)
    """
    ref_chains = polymer_chains(ref_model)
    gen_chains = polymer_chains(gen_model)
    if len(ref_chains) != len(gen_chains):
        raise ValueError(
            f"# polymer chains differ: ref={len(ref_chains)} gen={len(gen_chains)}"
        )

    # Precompute sequences & CA coords
    ref_seqs = [chain_seq_1letter(ch) for ch in ref_chains]
    gen_seqs = [chain_seq_1letter(ch) for ch in gen_chains]
    ref_coords = [extract_ca_coords(ch) for ch in ref_chains]
    gen_coords = [extract_ca_coords(ch) for ch in gen_chains]

    # Group indices by sequence
    ref_groups = defaultdict(list)
    gen_groups = defaultdict(list)
    for j, s in enumerate(ref_seqs):
        ref_groups[s].append(j)
    for i, s in enumerate(gen_seqs):
        gen_groups[s].append(i)

    BIG = 1e9
    best_rmsd = float("inf")
    best_assignment: Dict[int, int] = {}

    # Try anchors: for each shared sequence group, anchor ONE generated chain to EACH ref chain in that group
    shared = sorted(set(ref_groups.keys()) & set(gen_groups.keys()))
    if len(shared) == 0:
        print("Can not find identical sequences!")
        rename_map = {
            gen_chains[i].id: gen_chains[i].id for i in range(len(gen_chains))
        }
        return rename_map, 0.0

    for s in shared:
        gen_rep = gen_groups[s][
            0
        ]  # representative generated-chain index for this sequence
        for j in ref_groups[s]:
            # Fit transform using the anchor pair
            rot, tr_p, tr_q = fit_rotran_from_pair(gen_coords[gen_rep], ref_coords[j])

            # Build cost matrix under fixed (rot, tran); restrict to same-sequence pairs
            m, n = len(gen_chains), len(ref_chains)
            cost = np.full((m, n), BIG, dtype=float)
            for gi in range(m):
                Gi = apply_rotran(gen_coords[gi], rot, tr_p, tr_q)
                sgi = gen_seqs[gi]
                for rj in ref_groups.get(sgi, []):  # only same-seq columns
                    Rj = ref_coords[rj]
                    nn = min(len(Gi), len(Rj))
                    if nn == 0:  # no CA overlap
                        continue
                    diff = Gi[:nn] - Rj[:nn]
                    cost[gi, rj] = float(np.sqrt((diff * diff).sum() / nn))

            # Force the anchor pair (gen_rep -> j)
            cost[gen_rep, :] = BIG
            cost[:, j] = BIG
            cost[gen_rep, j] = 0.0

            # Solve assignment and evaluate complex RMSD
            row_ind, col_ind = linear_sum_assignment(cost)
            mapping_idx = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
            rmsd = complex_rmsd_under_transform(
                gen_coords, ref_coords, mapping_idx, rot, tr_p, tr_q
            )
            print(mapping_idx, rmsd)

            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_assignment = mapping_idx

    if not best_assignment:
        raise RuntimeError("No valid anchored mapping found. Check sequences/CA atoms.")

    # Convert index mapping to chain-id mapping
    rename_map = {
        gen_chains[i].id: ref_chains[j].id for i, j in best_assignment.items()
    }
    return rename_map, best_rmsd


# ------------------------- writing outputs -------------------------


def read_entry_id_from_cif_text(path: str) -> str:
    """Try to read true mmCIF _entry.id; fallback to filename stem."""
    try:
        with open(path, "r") as f:
            for line in f:
                if line.startswith("_entry.id"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]
    except Exception:
        pass
    return os.path.splitext(os.path.basename(path))[0]


def write_cif_with_mapping(
    generated_cif: str, rename_map: Dict[str, str], out_cif: str
):
    """Rename chain IDs in a Biotite AtomArray and write mmCIF, preserving entry_id."""
    parser = MMCIFParser(generated_cif)
    atom_array = parser.get_structure(
        altloc="first", model=1, bond_lenth_threshold=None
    )

    # Try to preserve original entry_id
    entry_id = getattr(parser, "entry_id", None) or read_entry_id_from_cif_text(
        generated_cif
    )

    # Rename chains & reorder by target IDs for neatness
    new_chain_id = atom_array.chain_id.copy()
    for old_id, new_id in rename_map.items():
        mask = atom_array.chain_id == old_id
        new_chain_id[mask] = new_id
    atom_array.chain_id = new_chain_id

    desired_order = sorted(set(rename_map.values()))
    idxs = []
    for cid in desired_order:
        idx = np.where(atom_array.chain_id == cid)[0]
        idxs.extend(idx)
    if idxs:
        atom_array = atom_array[np.array(idxs, dtype=int)]

    writer = CIFWriter(atom_array=atom_array, entity_poly_type=parser.entity_poly_type)
    writer.save_to_cif(out_cif, entry_id=entry_id, include_bonds=True)


def copy_residue(res_src: Residue) -> Residue:
    """Deep-copy a Biopython Residue (including atoms), preserving id & resname."""
    new_res = Residue(res_src.id, res_src.get_resname(), "")
    serial = 1
    for atom in res_src:
        name = atom.get_name()
        coord = atom.get_coord()
        bfactor = atom.get_bfactor()
        occ = atom.get_occupancy() if atom.get_occupancy() is not None else 1.0
        altloc = atom.get_altloc() if atom.get_altloc() else " "
        fullname = atom.get_fullname()
        element = atom.element or (name[0].upper())
        new_atom = Atom(
            name, coord, bfactor, occ, altloc, fullname, serial, element.strip()
        )
        new_res.add(new_atom)
        serial += 1
    return new_res


def write_pdb_with_mapping(
    generated_pdb: str, rename_map: Dict[str, str], out_pdb: str
):
    """
    Rebuild a new PDB where chains are renamed per rename_map and
    written in alphabetical order of the *destination* chain IDs.
    Rebuilding avoids in-place ID collisions when chains swap names.
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("gen", generated_pdb)
    model = next(struct.get_models())

    # Collect residues per destination chain id (dst_id)
    # If multiple source chains map to the same dst_id, we append their residues in source order.
    dst_residues = {}  # dst_id -> list[Residue(copy)]
    for src_chain in list(model):
        src_id = src_chain.id
        dst_id = rename_map.get(src_id, src_id)
        if dst_id not in dst_residues:
            dst_residues[dst_id] = []
        for res in src_chain:
            dst_residues[dst_id].append(copy_residue(res))

    # Build new structure with chains in alphabetical order
    new_struct = Structure("renamed")
    new_model = Model(0)
    new_struct.add(new_model)

    for dst_id in sorted(dst_residues.keys()):
        ch = Chain(dst_id)
        for res in dst_residues[dst_id]:
            ch.add(res)
        new_model.add(ch)

    io = PDBIO()
    io.set_structure(new_struct)
    io.save(out_pdb)


# ------------------------- main orchestration -------------------------


def permute_generated_min_complex_rmsd(
    generated_path: str,
    reference_path: str,
    out_path: str,
) -> float:
    """
    Support any mix of PDB/mmCIF for (generated, reference).
    - Compute chain rename map with anchored reduced enumeration.
    - Write output in the same format as the generated input.
    Returns: best complex CA RMSD (float).
    """
    # Load models for RMSD / mapping
    ref_struct, ref_model = load_biopython_model(reference_path)
    gen_struct, gen_model = load_biopython_model(generated_path)

    # Build mapping (gen chain id -> ref chain id)
    rename_map, best_rmsd = anchored_optimal_mapping_grouped(ref_model, gen_model)
    if all([k == v for k, v in rename_map.items()]) and generated_path == out_path:
        print("[INFO] No need to perform chain permutation, skip!")
        return best_rmsd

    # Write output in the same format as the generated input
    if is_cif(generated_path):
        write_cif_with_mapping(generated_path, rename_map, out_path)
    elif is_pdb(generated_path):
        write_pdb_with_mapping(generated_path, rename_map, out_path)
    else:
        raise ValueError("Unsupported format for generated file.")

    print(f"[INFO] Chain mapping (generated → reference): {rename_map}")
    print(f"[INFO] Best complex CA RMSD (anchored scheme): {best_rmsd:.4f} Å")
    return best_rmsd
