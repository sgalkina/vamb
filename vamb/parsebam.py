# This script calculates abundances of contigs in BAM files.

# Theory:
# We want a simple way to estimate abundance of redundant contig catalogues.
# Earlier we used to run analysis on deduplicated gene catalogues, but since
# both depth and kmer composition are only stable for longer contigs, we have
# moved to contig catalogues. We have not found a way of deduplicating contigs.

# Earlier versions simply counted the number of reads per reference, but we
# found that was not accurate enough, presumably because aligners have a hard time
# properly aligning to end of contigs.
# This script calculates the mean depth across contigs, excluding one read length
# (though max 100 bp) of each end. It normalizes by contig length, number of reads,
# and mean read length.

# We look at all hits, including secondary hits. We do not discount partial
# alignments. Also, if a read maps to N contigs, we count each hit as 1/N reads.
# The reason for all these decisions is that if the aligner believes it's a hit,
# we believe the contig is present.
# We count each read independently, because some algners assigns mating reads
# to different contigs independently

__doc__ = """Estimate abundances from BAM files of reads mapped to contigs.

Usage:
>>> bampaths = ['/path/to/bam1.bam', '/path/to/bam2.bam', '/path/to/bam3.bam']
>>> rpkms = read_bamfiles(bampaths)
"""

import pysam as _pysam

import sys as _sys
import os as _os
import multiprocessing as _multiprocessing
import numpy as _np
import time as _time
from hashlib import md5 as _md5

DEFAULT_SUBPROCESSES = min(8, _os.cpu_count())

def mergecolumns(pathlist):
    """Merges multiple npz files with columns to a matrix.

    All paths must be npz arrays with the array saved as name 'arr_0',
    and with the same length.

    Input: pathlist: List of paths to find .npz files to merge
    Output: Matrix with one column per npz file
    """

    if len(pathlist) == 0:
        return _np.array([], dtype=_np.float32)

    for path in pathlist:
        if not _os.path.exists(path):
            raise FileNotFoundError(path)

    first = _np.load(pathlist[0])['arr_0']
    length = len(first)
    ncolumns = len(pathlist)

    result = _np.zeros((length, ncolumns), dtype=_np.float32)
    result[:,0] = first

    for columnno, path in enumerate(pathlist[1:]):
        column = _np.load(path)['arr_0']
        if len(column) != length:
            raise ValueError("Length of data at {} is not equal to that of "
                             "{}".format(path, pathlist[0]))
        result[:,columnno + 1] = column

    return result

def _mean_read_length(file, N, default=100):
    "Get (mean read length, edge_adjustment) of first N reads of alignment file"
    lengths = list()
    for segment in file:
        if len(lengths) == N:
            break

        qlen = segment.infer_query_length()
        # This can happen if there is no CIGAR
        if qlen is not None:
            lengths.append(qlen)

    if len(lengths) == 0:
        mean_length = default
    else:
        mean_length = int(sum(lengths) / len(lengths))

    file.reset() # go back to beginning of file
    return mean_length, min(mean_length, 100)

def _matches_mismatches(segment):
    """Return (matches, mismatches, total_mm) of the given aligned segment.
    Insertions/deletions counts as mismatches. total_mm includes ins/del"""
    matches, other = 0, 0
    for (operation, n) in segment.cigartuples:
        if operation in (0, 7, 8): # mm / matchers / mismatches
            matches += n
        elif operation in (1, 2): # insertion/deletion
            other += n

    mismatches = segment.get_tag('NM')
    return matches - mismatches, mismatches, other + mismatches

def _count_covered_bases(segment, mean_read_length, reflen, matches):
    "Counts all matches/mm not within the first or last 75 bp of the reference."
    start = segment.reference_start
    end = segment.reference_end

    # If the read does not overlap with the ends, just return all matches
    # in read, this is simpler and faster
    if start >= mean_read_length and end < reflen - mean_read_length:
        return matches

    matches = 0
    pos = start
    started = False

    # Else count all matches that does not overlap with the ends. BAM files report
    # the "starting" position as the start of the alignment proper, and so we must
    # skip every cigar operation that does not consume reference bases, otherwise
    # the "pos" variable will be wrong.
    for (kind, n) in segment.cigartuples:
        if not started:
            if kind in (1, 4, 5, 6): # insertion, soft clip, hard clip, padding
                continue
            else:
                started = True

        # Actual matching operations, discard first and last 75 bp of reference
        if kind in (0, 7, 8):
            matches += max(0, min((pos+n), reflen-mean_read_length) - max(pos, mean_read_length))
        pos += n

    return matches

    return False

def _is_skippable(segment, minscore, identity, minid):
    "Return whether this segment should be ignored when calculating depth"
    # Skip if unaligned or suppl. aligment
    if segment.flag & 0x804 != 0:
        return True

    elif minscore is not None and alignedsegment.get_tag('AS') < minscore:
        return True

    if minid is not None and identity < minid:
        return True

    return False

def _group_segments(segmentiterator, minscore, minid):
    """Returns an iterator of ([(segment, matches) ...], nreads), such that nreads
    counts all distinct reads in the file, and each returned list comprises
    all the segments from a single read, as well as the matches."""
    readcount = 0
    buffer = list()

    try:
        segment = next(segmentiterator)
        old_read = (segment.is_reverse, segment.query_name)
        readcount += 1
        matches, mismatches, total = _matches_mismatches(segment)
        identity = matches / (matches + total)
        if not _is_skippable(segment, minscore, identity, minid):
            buffer.append((segment, matches))

    except StopIteration:
        raise StopIteration from None

    for segment in segmentiterator:
        read = (segment.is_reverse, segment.query_name)
        if read != old_read:
            readcount += 1

            if len(buffer) != 0:
                yield (buffer, readcount)
                buffer = list()

            read = old_read

        matches, mismatches, total = _matches_mismatches(segment)
        identity = matches / (matches + total)
        if not _is_skippable(segment, minscore, identity, minid):
            buffer.append((segment, matches))

    # Final read
    if len(buffer) != 0:
        yield (buffer, readcount)

def _calc_covered_bases(bamfile, mean_read_length, minscore=None, minid=None):
    """Count number of reads mapping to each reference in a bamfile,
    optionally filtering for score and minimum id.
    Multi-mapping reads MUST be consecutive in file, and their counts are
    split among the references.

    Inputs:
        bamfile: Open pysam.AlignmentFile
        mean_read_length: Mean read length of file
        minscore: Minimum alignment score (AS field) to consider [None]
        minid: Discard any reads with ID lower than this [None]

    Output:
        depths: Float32 Numpy array of read counts for each reference in file.
        readcount: Number of distinct mapped reads
    """
    # Use 64-bit floats for better precision when counting
    lengths = bamfile.lengths
    depths = _np.zeros(len(lengths))
    nreads = 0

    for segments, nreads in _group_segments(bamfile, minscore, minid):
        fraction = 1.0 / len(segments)
        for (segment, matches) in segments:
            bases = _count_covered_bases(segment, mean_read_length, lengths[segment.reference_id], matches)
            depths[segment.reference_id] += bases * fraction

    return depths.astype(_np.float32), nreads

def calc_adjusted_depths(bases, lengths, readcount, mean_read_length, edge_adjustment, minlength=151):
    """Calculate depths adjusted by number of reads and seq lengths

    Inputs:
        bases: Numpy vector from _calc_covered_bases
        lengths: Iterable of contig lengths in same order as counts
        readcount: Number of mapped reads in total
        mean_read_length: Read length average
        minlength [151]: Discard any references shorter than N bases

    Output: Float32 Numpy vector of RPKM for all seqs with length >= minlength
    """
    if len(bases) != len(lengths):
        raise ValueError("bases length and lengths length must be same")

    if minlength is not None and minlength <= edge_adjustment:
        raise ValueError("Minlength must be larger than edge adjustment")

    # Compensate for one mean read length of each end where reads don't align properly,
    # we don't count depth at these positions
    lengtharray =  _np.array(lengths)
    lengtharray[lengtharray > (2*edge_adjustment)] -= (2*edge_adjustment) # prevent division by zero

    adjusted_depths = (bases / lengtharray).astype(_np.float32)

    # Scale with million reads, since depth is a linear function of total number
    # of reads. Also scale with read length to compare samples with differing
    # read lengths.
    if readcount > 0:
        adjusted_depths *= 1000000 / (readcount * mean_read_length)

    # Now filter away small contigs
    adjusted_depths = adjusted_depths[lengtharray >= minlength]

    return adjusted_depths

def _hash_refnames(refnames):
    "Hashes an iterable of strings of reference names using MD5."
    hasher = _md5()
    for refname in refnames:
        hasher.update(refname.encode().rstrip())

    return hasher.digest()

def _check_bamfile(path, bamfile, refhash, minlength):
    "Checks bam file for correctness (refhash and sort order). To be used before parsing."
    # If refhash is set, check ref hash matches what is found.
    if refhash is not None:
        if minlength is None:
            refnames = bamfile.references
        else:
            pairs = zip(bamfile.references, bamfile.lengths)
            refnames = (ref for (ref, len) in pairs if len >= minlength)

        hash = _hash_refnames(refnames)
        if hash != refhash:
            errormsg = ('BAM file {} has reference hash {}, expected {}. '
                        'Verify that all BAM headers and FASTA headers are '
                        'identical and in the same order.')
            raise ValueError(errormsg.format(path, hash.hex(), refhash.hex()))

    # Check that file is unsorted or sorted by read name.
    hd_header = bamfile.header.get("HD", dict())
    sort_order = hd_header.get("SO")
    if sort_order in ("coordinate", "unknown"):
        errormsg = ("BAM file {} is marked with sort order '{}', must be "
                    "unsorted or sorted by readname.")
        raise ValueError(errormsg.format(path, sort_order))


def _get_contig_rpkms(inpath, outpath, refhash, minscore, minlength, minid):
    """Returns  RPKM (reads per kilobase per million mapped reads)
    for all contigs present in BAM header.

    Inputs:
        inpath: Path to BAM file
        outpath: Path to dump depths array to or None
        refhash: Expected reference hash (None = no check)
        minscore: Minimum alignment score (AS field) to consider
        minlength: Discard any references shorter than N bases
        minid: Discard any reads with ID lower than this

    Outputs:
        path: Same as input path
        rpkms:
            If outpath is not None: None
            Else: A float32-array with RPKM for each contig in BAM header
        length: Length of rpkms array
    """

    bamfile = _pysam.AlignmentFile(inpath, "rb")
    _check_bamfile(inpath, bamfile, refhash, minlength)
    mean_read_length, edge_adjustment = _mean_read_length(bamfile, 1000)
    bases, readcount = _calc_covered_bases(bamfile, edge_adjustment, minscore, minid)
    adjusted_depths = calc_adjusted_depths(bases, bamfile.lengths, readcount,
                      mean_read_length, edge_adjustment, minlength)
    bamfile.close()

    # If dump to disk, array returned is None instead of rpkm array
    if outpath is not None:
        arrayresult = None
        _np.savez_compressed(outpath, adjusted_depths)
    else:
        arrayresult = adjusted_depths

    return inpath, arrayresult, len(adjusted_depths)

def read_bamfiles(paths, dumpdirectory=None, refhash=None, minscore=None, minlength=200,
                  minid=None, subprocesses=DEFAULT_SUBPROCESSES, logfile=None):
    "Placeholder docstring - replaced after this func definition"

    # Define callback function depending on whether a logfile exists or not
    if logfile is not None:
        def _callback(result):
            path, rpkms, length = result
            print('\tProcessed', path, file=logfile)
            logfile.flush()

    else:
        def _callback(result):
            pass

    # Bam files must be unique.
    if len(paths) != len(set(paths)):
        raise ValueError('All paths to BAM files must be unique.')

    # Bam files must exist
    for path in paths:
        if not _os.path.isfile(path):
            raise FileNotFoundError(path)

    if dumpdirectory is not None:
        # Dumpdirectory cannot exist, but its parent must exist
        dumpdirectory = _os.path.abspath(dumpdirectory)
        if _os.path.exists(dumpdirectory):
            raise FileExistsError(dumpdirectory)

        parentdir = _os.path.dirname(_os.path.abspath(dumpdirectory))
        if not _os.path.isdir(parentdir):
            raise FileNotFoundError("Parent dir of " + dumpdirectory)

        # Create directory to dump in
        _os.mkdir(dumpdirectory)

    # Spawn independent processes to calculate RPKM for each of the BAM files
    processresults = list()

    # Queue all the processes
    with _multiprocessing.Pool(processes=subprocesses) as pool:
        for pathnumber, path in enumerate(paths):
            if dumpdirectory is None:
                outpath = None
            else:
                outpath = _os.path.join(dumpdirectory, str(pathnumber) + '.npz')

            arguments = (path, outpath, refhash, minscore, minlength, minid)
            processresults.append(pool.apply_async(_get_contig_rpkms, arguments,
                                                   callback=_callback))

        all_done, any_fail = False, False
        while not (all_done or any_fail):
            _time.sleep(5)
            all_done = all(process.ready() and process.successful() for process in processresults)
            any_fail = any(process.ready() and not process.successful() for process in processresults)

            if all_done:
                pool.close() # exit gently
            if any_fail:
                pool.terminate() # exit less gently

        # Wait for all processes to be cleaned up
        pool.join()

    # Raise the error if one of them failed.
    for path, process in zip(paths, processresults):
        if process.ready() and not process.successful():
            print('\tERROR WHEN PROCESSING:', path, file=logfile)
            print('Vamb aborted due to error in subprocess. See stacktrace for source of exception.')
            if logfile is not None:
                logfile.flush()
            process.get()

    ncontigs = None
    for processresult in processresults:
        path, rpkm, length = processresult.get()

        # Verify length of contigs are same for all BAM files
        if ncontigs is None:
            ncontigs = length
        elif length != ncontigs:
            raise ValueError('First BAM file has {} headers, {} has {}.'.format(
                             ncontigs, path, length))

    # If we did not dump to disk, load directly from process results to
    # one big matrix...
    if dumpdirectory is None:
        columnof = {p:i for i, p in enumerate(paths)}
        rpkms = _np.zeros((ncontigs, len(paths)), dtype=_np.float32)

        for processresult in processresults:
            path, rpkm, length = processresult.get()
            rpkms[:, columnof[path]] = rpkm

    # If we did, instead merge them from the disk
    else:
        dumppaths = [_os.path.join(dumpdirectory, str(i) + '.npz') for i in range(len(paths))]
        rpkms = mergecolumns(dumppaths)

    return rpkms

read_bamfiles.__doc__ = """Spawns processes to parse BAM files and get contig rpkms.

Input:
    path: List or tuple of paths to BAM files
    dumpdirectory: [None] Dir to create and dump per-sample depths NPZ files to
    refhash: [None]: Check all BAM references md5-hash to this (None = no check)
    minscore [None]: Minimum alignment score (AS field) to consider
    minlength [200]: Ignore any references shorter than N bases
    minid [None]: Discard any reads with nucleotide identity less than this
    subprocesses [{}]: Number of subprocesses to spawn
    logfile: [None] File to print progress to

Output: A (n_contigs x n_samples) Numpy array with RPKM
""".format(DEFAULT_SUBPROCESSES)
