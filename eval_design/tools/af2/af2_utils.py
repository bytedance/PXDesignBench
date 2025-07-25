"""
This function implements a cyclic offset matrix for connecting the N- and C-termini
of sequences (e.g., cyclic peptides), adapted from ColabDesign's cyclic peptide design.

Reference:
- ColabDesign GitHub Repository:
  https://github.com/sokrypton/ColabDesign/blob/main/af/examples/af_cyc_design.ipynb
"""

import numpy as np


def add_cyclic_offset(self, offset_type=2):
    """add cyclic offset to connect N and C term"""

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
