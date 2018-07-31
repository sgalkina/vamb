
__doc__ = """Calculate z-normalized tetranucleotide frequency from a FASTA file.

Usage:
>>> with open('/path/to/contigs.fna', 'rb') as filehandle
...     tnfs, contignames, lengths = read_contigs(filehandle)
"""

__cmd_doc__ = """Calculate z-normalized tetranucleotide frequency from a FASTA file.

Input: A fasta file containing only A, C, G, T and N sequences
Output: A (n_sequences * 136) matrix of tetranucleotide frequencies,
z-normalized through axis 0.
"""



import sys as _sys
import os as _os
import numpy as _np
import gzip as _gzip
import vamb.vambtools as _vambtools
    
if __name__ == '__main__':
    import argparse



TNF_HEADER = '#contigheader\t' + '\t'.join([
'AAAA/TTTT', 'AAAC/GTTT', 'AAAG/CTTT', 'AAAT/ATTT',
'AACA/TGTT', 'AACC/GGTT', 'AACG/CGTT', 'AACT/AGTT',
'AAGA/TCTT', 'AAGC/GCTT', 'AAGG/CCTT', 'AAGT/ACTT',
'AATA/TATT', 'AATC/GATT', 'AATG/CATT', 'AATT',
'ACAA/TTGT', 'ACAC/GTGT', 'ACAG/CTGT', 'ACAT/ATGT',
'ACCA/TGGT', 'ACCC/GGGT', 'ACCG/CGGT', 'ACCT/AGGT',
'ACGA/TCGT', 'ACGC/GCGT', 'ACGG/CCGT', 'ACGT',
'ACTA/TAGT', 'ACTC/GAGT', 'ACTG/CAGT', 'AGAA/TTCT',
'AGAC/GTCT', 'AGAG/CTCT', 'AGAT/ATCT', 'AGCA/TGCT',
'AGCC/GGCT', 'AGCG/CGCT', 'AGCT',      'AGGA/TCCT',
'AGGC/GCCT', 'AGGG/CCCT', 'AGTA/TACT', 'AGTC/GACT',
'AGTG/CACT', 'ATAA/TTAT', 'ATAC/GTAT', 'ATAG/CTAT',
'ATAT',      'ATCA/TGAT', 'ATCC/GGAT', 'ATCG/CGAT',
'ATGA/TCAT', 'ATGC/GCAT', 'ATGG/CCAT', 'ATTA/TAAT',
'ATTC/GAAT', 'ATTG/CAAT', 'CAAA/TTTG', 'CAAC/GTTG',
'CAAG/CTTG', 'CACA/TGTG', 'CACC/GGTG', 'CACG/CGTG',
'CAGA/TCTG', 'CAGC/GCTG', 'CAGG/CCTG', 'CATA/TATG',
'CATC/GATG', 'CATG',      'CCAA/TTGG', 'CCAC/GTGG',
'CCAG/CTGG', 'CCCA/TGGG', 'CCCC/GGGG', 'CCCG/CGGG',
'CCGA/TCGG', 'CCGC/GCGG', 'CCGG',      'CCTA/TAGG',
'CCTC/GAGG', 'CGAA/TTCG', 'CGAC/GTCG', 'CGAG/CTCG',
'CGCA/TGCG', 'CGCC/GGCG', 'CGCG',      'CGGA/TCCG',
'CGGC/GCCG', 'CGTA/TACG', 'CGTC/GACG', 'CTAA/TTAG',
'CTAC/GTAG', 'CTAG',      'CTCA/TGAG', 'CTCC/GGAG',
'CTGA/TCAG', 'CTGC/GCAG', 'CTTA/TAAG', 'CTTC/GAAG',
'GAAA/TTTC', 'GAAC/GTTC', 'GACA/TGTC', 'GACC/GGTC',
'GAGA/TCTC', 'GAGC/GCTC', 'GATA/TATC', 'GATC',
'GCAA/TTGC', 'GCAC/GTGC', 'GCCA/TGGC', 'GCCC/GGGC',
'GCGA/TCGC', 'GCGC',      'GCTA/TAGC', 'GGAA/TTCC',
'GGAC/GTCC', 'GGCA/TGCC', 'GGCC', 'GGGA/TCCC',
'GGTA/TACC', 'GTAA/TTAC', 'GTAC', 'GTCA/TGAC',
'GTGA/TCAC', 'GTTA/TAAC', 'TAAA/TTTA', 'TACA/TGTA',
'TAGA/TCTA', 'TATA',      'TCAA/TTGA', 'TCCA/TGGA',
'TCGA',      'TGAA/TTCA', 'TGCA', 'TTAA']) + '\n'



def read_contigs(byte_iterator, minlength=100):
    """Parses a FASTA file open in binary reading mode.
    
    Input:
        byte_iterator: Iterator of binary lines of a FASTA file
        minlength[100]: Ignore any references shorter than N bases 
    
    Outputs:
        tnfs: A (n_FASTA_entries x 136) matrix of tetranucleotide freq.
        contignames: A list of contig headers
        lengths: A Numpy array of contig lengths
    """
    
    tnf_list = list()
    contignames = list()
    lengths = list()
    
    if byte_iterator is not iter(byte_iterator):
        raise ValueError('byte_iterator is not an iterator')
        
    entries = _vambtools.byte_iterfasta(byte_iterator)

    for entry in entries:
        if len(entry) < minlength:
            continue

        tnf_list.append(entry.fourmer_freq())
        contignames.append(entry.header)
        lengths.append(len(entry))
    
    lengths = _np.array(lengths, dtype=_np.int)
    
    tnfs = _np.array(tnf_list, dtype=_np.float32)
    del tnf_list
    
    if tnfs.ndim < 2 or tnfs.shape[1] < 2:
        raise IndexError('Cannot normalize less than 2 contigs')
    
    _vambtools.zscore(tnfs, axis=0, inplace=True)
    
    return tnfs, contignames, lengths

