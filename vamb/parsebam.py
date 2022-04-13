__doc__ = """Estimate depths from BAM files of reads mapped to contigs.

Usage:
>>> bampaths = ['/path/to/bam1.bam', '/path/to/bam2.bam', '/path/to/bam3.bam']
>>> rpkms = read_bamfiles(bampaths)
"""

import pycoverm as _pycoverm
import os as _os
import numpy as _np
from vamb.parsecontigs import CompositionMetaData
import vamb.vambtools as _vambtools
from typing import Optional, TypeVar, Union, BinaryIO

_ncpu = _os.cpu_count()
DEFAULT_THREADS = 8 if _ncpu is None else _ncpu

A = TypeVar('A', bound='Abundance')


class Abundance:
    "Object representing contig abundance. Contains a matrix and refhash."

    __slots__ = ['matrix', 'refhash']

    def __init__(self, matrix: _np.ndarray, refhash: bytes):
        assert matrix.dtype == _np.float32
        assert matrix.ndim == 2

        self.matrix = matrix
        self.refhash = refhash

    @property
    def nseqs(self) -> int:
        return len(self.matrix)

    def verify_refhash(self, refhash: bytes) -> None:
        if self.refhash != refhash:
            raise ValueError(
                f"BAM files reference name hash to {self.refhash.hex()}, "
                f"expected {refhash.hex()}. "
                "Make sure all BAM and FASTA headers are identical "
                "and in the same order."
            )

    def save(self, io: Union[str, BinaryIO]):
        _np.savez_compressed(io, matrix=self.matrix, refhash=self.refhash)

    @classmethod
    def load(
        cls: type[A],
        io: Union[str, BinaryIO],
        refhash: Optional[bytes]
    ) -> A:
        arrs = _np.load(io, allow_pickle=True)
        abundance = cls(_vambtools.validate_input_array(
            arrs['matrix']), arrs["refhash"].item())
        if refhash is not None:
            abundance.verify_refhash(refhash)

        return abundance

    @classmethod
    def from_files(
        cls: type[A],
        paths: list[str],
        comp_metadata: CompositionMetaData,
        verify_refhash: bool,
        minid: float,
        nthreads: int
    ) -> A:
        """Input:
        paths: List of paths to BAM files
        comp_metadata: CompositionMetaData of sequence catalogue used to make BAM files
        verify_refhash: Whether to verify composition and BAM references are the same
        minid: Discard any reads with nucleotide identity less than this
        nthreads: Use this number of threads for coverage estimation
        """
        if minid < 0 or minid > 1:
            raise ValueError(f"minid must be between 0 and 1, not {minid}")

        for path in paths:
            if not _os.path.isfile(path):
                raise FileNotFoundError(path)

            if not _pycoverm.is_bam_sorted(path):
                raise ValueError(f"Path {path} is not sorted by reference.")

        headers, coverage = _pycoverm.get_coverages_from_bam(
            paths, threads=nthreads, min_identity=minid,
            # Note: pycoverm's trim_upper=0.1 is same as CoverM trim-upper 90.
            trim_upper=0.1, trim_lower=0.1
        )

        assert len(headers) == len(coverage)
        assert coverage.shape[1] == len(paths)

        # Filter length, using comp_metadata's mask, which has been set by minlength
        if len(comp_metadata.mask) != len(headers):
            raise ValueError(
                f"CompositionMetaData was created with {len(comp_metadata.mask)} sequences, "
                f"but number of refs in BAM files are {len(headers)}."
            )

        headers = [h for (h, m) in zip(headers, comp_metadata.mask) if m]
        _vambtools.numpy_inplace_maskarray(coverage, comp_metadata.mask)

        refhash = _vambtools.hash_refnames(headers)
        abundance = cls(coverage, refhash)

        # Check refhash
        if verify_refhash:
            abundance.verify_refhash(comp_metadata.refhash)

        return abundance
