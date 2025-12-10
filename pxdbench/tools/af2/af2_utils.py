import numpy as np
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure


def add_cyclic_offset(self, offset_type=2):
    """
    This function implements a cyclic offset matrix for connecting the N- and C-termini
    of sequences (e.g., cyclic peptides), adapted from ColabDesign's cyclic peptide design.

    Reference:
    - ColabDesign GitHub Repository:
    https://github.com/sokrypton/ColabDesign/blob/main/af/examples/af_cyc_design.ipynb
    """

    def cyclic_offset(L):
        i = np.arange(L)
        ij = np.stack([i, i + L], -1)
        offset = i[:, None] - i[None, :]
        c_offset = np.abs(ij[:, None, :, None] - ij[None, :, None, :]).min((2, 3))
        if offset_type == 1:
            c_offset = c_offset
        elif offset_type >= 2:
            a = c_offset < np.abs(offset)
            c_offset[a] = -c_offset[a]
        if offset_type == 3:
            idx = np.abs(c_offset) > 2
            c_offset[idx] = (32 * c_offset[idx]) / abs(c_offset[idx])
        return c_offset * np.sign(offset)

    idx = self._inputs["residue_index"]
    offset = np.array(idx[:, None] - idx[None, :])
    if self.protocol == "binder":
        c_offset = cyclic_offset(self._binder_len)
        offset[self._target_len :, self._target_len :] = c_offset
    if self.protocol in ["fixbb", "partial", "hallucination"]:
        Ln = 0
        for L in self._lengths:
            offset[Ln : Ln + L, Ln : Ln + L] = cyclic_offset(L)
            Ln += L
    self._inputs["offset"] = offset


def extract_labels(model):
    """
    Extract residue labels from the reference model in sequential order.
    Each label is a tuple: (chain_id, hetflag, resseq:int, icode:str)
    - chain_id: chain identifier
    - hetflag: ' ' for standard residues, or HETATM flag for hetero groups
    - resseq: residue sequence number
    - icode: insertion code (default ' ' if empty)
    """
    labels = []
    for chain in model:
        for res in chain:
            chain_id = chain.id
            hetflag, resseq, icode = res.id
            labels.append((chain_id, hetflag, int(resseq), icode if icode else " "))
    return labels


def residues_by_chain(model):
    """Group standard residues by chain, preserving original chain order."""
    d = {}
    for ch in model:
        bucket = []
        for res in ch:
            het, _, _ = res.id
            if het != " ":
                continue
            if res.get_resname() == "HOH":
                continue
            bucket.append(res)
        d[ch.id] = bucket
    return d


def copy_residue_with_new_id(src_res, new_id):
    """
    Create a copy of a residue with a new ID.
    - new_id: tuple (hetflag, resseq, icode)
    - Copies all atoms from the source residue.
    """
    hetflag, resseq, icode = new_id
    new_res = Residue(new_id, src_res.get_resname(), "")
    serial = 1
    for atom in src_res:
        name = atom.get_name()
        coord = atom.get_coord()
        bfactor = atom.get_bfactor()
        occ = atom.get_occupancy() if atom.get_occupancy() is not None else 1.0
        altloc = atom.get_altloc() if atom.get_altloc() else " "
        fullname = atom.get_fullname()  # atom name in PDB format
        element = atom.element or (name[0].upper())
        new_atom = Atom(
            name, coord, bfactor, occ, altloc, fullname, serial, element.strip()
        )
        new_res.add(new_atom)
        serial += 1
    return new_res


def renumber_by_rebuilding(
    pdb1_ref_path: str,
    pdb2_in_path: str,
    out_path: str,
    *,
    # layout of condition/binder in ref and tgt:
    #   "cond_first"  -> [condition chains..., binder chains...]
    #   "cond_last"   -> [binder chains..., condition chains...]
    ref_layout: str = "cond_last",
    tgt_layout: str = "cond_first",
    # identify binder chains in each file (remaining chains are treated as condition)
    # colabdesign AF2 use chain B as the binder chain
    binder_chain_ids_tgt=("B",),
    # sanity guard: require equal residue counts (standard residues only)
    strict_len_check: bool = True,
):
    """
    Rebuild PDB2 using PDB1 as the numbering reference, while accommodating
    different chain layouts (condition-first vs condition-last).

    Strategy:
      1) Take residue label stream from REF (chain order + resseq/icode).
      2) Reorder TGT residues by groups (condition vs binder) to match the REF layout.
      3) Zip(ref_labels, reordered_tgt_residues) and rebuild.

    Assumptions:
      - Standard protein residues only (HETATM/HOH skipped on both sides).
      - Binder chains are identified by 'binder_chain_ids_*'; all other chains are 'condition'.
      - Within each group (binder/condition), original chain order is preserved.
    """
    parser = PDBParser(QUIET=True)
    ref_struct = parser.get_structure("ref", pdb1_ref_path)
    tgt_struct = parser.get_structure("tgt", pdb2_in_path)

    ref_model = next(ref_struct.get_models())
    tgt_model = next(tgt_struct.get_models())

    # 1) Reference labels (drives the final (chain, resseq, icode))
    labels = extract_labels(ref_model)

    # 2) Build target residue stream reordered to match ref layout semantics
    #    Partition target chains into binder vs condition using provided IDs.
    binder_set_tgt = set(binder_chain_ids_tgt)
    by_chain_tgt = residues_by_chain(tgt_model)

    # preserve the original chain order within each group
    binder_chains_tgt = [cid for cid in by_chain_tgt.keys() if cid in binder_set_tgt]
    cond_chains_tgt = [cid for cid in by_chain_tgt.keys() if cid not in binder_set_tgt]

    def flatten_chain_list(chain_ids):
        seq = []
        for cid in chain_ids:
            seq.extend(by_chain_tgt.get(cid, []))
        return seq

    if ref_layout == "cond_first":
        # We need TGT residues in [condition..., binder...] order.
        tgt_stream = (
            flatten_chain_list(cond_chains_tgt) + flatten_chain_list(binder_chains_tgt)
            if tgt_layout == "cond_last"
            else flatten_chain_list(cond_chains_tgt)
            + flatten_chain_list(binder_chains_tgt)
        )
    elif ref_layout == "cond_last":
        # We need TGT residues in [binder..., condition...] order.
        tgt_stream = (
            flatten_chain_list(binder_chains_tgt) + flatten_chain_list(cond_chains_tgt)
            if tgt_layout == "cond_first"
            else flatten_chain_list(binder_chains_tgt)
            + flatten_chain_list(cond_chains_tgt)
        )
    else:
        raise ValueError(f"Unknown ref_layout: {ref_layout}")

    if strict_len_check and len(labels) != len(tgt_stream):
        raise ValueError(
            f"Residue count mismatch when renumbering: ref={len(labels)}, tgt={len(tgt_stream)}"
        )

    # 3) Rebuild a new structure following REF labels, filling residues from TGT stream
    new_struct = Structure("renumbered")
    new_model = Model(0)
    new_struct.add(new_model)

    chain_cache = {}
    for (chain_id, hetflag, resseq, icode), src_res in zip(labels, tgt_stream):
        if chain_id not in chain_cache:
            chain_cache[chain_id] = Chain(chain_id)
            new_model.add(chain_cache[chain_id])

        new_res = copy_residue_with_new_id(src_res, (hetflag, int(resseq), icode))
        chain_cache[chain_id].add(new_res)

    # Save the rebuilt structure
    io = PDBIO()
    io.set_structure(new_struct)
    io.save(out_path)
    print(f"[OK] Rebuilt and renumbered PDB saved to: {out_path}")
