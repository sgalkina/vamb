import io
import unittest
import random
import string
import numpy as np

import vamb
from vamb.parsecontigs import Composition, CompositionMetaData

class TestReadContigs(unittest.TestCase):
    records = []
    io = io.BytesIO()
    chars = string.ascii_letters + string.digits
    symbols = np.array(bytearray(b"acgtACGTnNywsdbK"))
    weights = np.array([0.12] * 8 + [0.005] * 8)

    @classmethod
    def setUpClass(cls):
        for i in range(random.randrange(1400, 1500)):
            name = (
                b">"
                + "".join(
                    random.choices(cls.chars, k=random.randrange(12, 20))
                ).encode()
                + b" "
                + "".join(
                    random.choices(cls.chars, k=random.randrange(1, 10))
                ).encode()
            )
            seq = bytearray(
                np.random.choice(
                    cls.symbols, p=cls.weights, size=random.randrange(400, 600)
                )
            )
            cls.records.append(vamb.vambtools.FastaEntry(name, seq))

        for i in cls.records:
            cls.io.write(i.format().encode())
            cls.io.write(b"\n")

    def setUp(self):
        self.io.seek(0)

    def test_unique_names(self):
        with self.assertRaises(ValueError):
            CompositionMetaData(
                np.array(["foo", "foo"], dtype=object),
                np.array([1000, 1000]),
                np.array([True, True], dtype=bool),
                1000
            )

    def test_filter_minlength(self):
        minlen = 500
        composition = Composition.from_file(self.io, minlength=450)
        md = composition.metadata
        hash1 = md.refhash
        composition.filter_min_length(minlen)

        hash2 = md.refhash
        self.assertNotEqual(hash1, hash2)
        self.assertEqual(len(md.identifiers), len(md.lengths))
        self.assertEqual(md.nseqs, md.mask.sum())
        self.assertLessEqual(minlen, composition.metadata.lengths.min(initial=minlen))
        self.assertEqual(len(md.mask), len(self.records))

        md.filter_min_length(minlen + 50)
        self.assertEqual(len(md.identifiers), len(md.lengths))
        self.assertEqual(md.nseqs, md.mask.sum())
        self.assertLessEqual(minlen, composition.metadata.lengths.min(initial=minlen + 50))
        self.assertEqual(len(md.mask), len(self.records))

        hash3 = md.refhash
        md.filter_min_length(minlen - 50)
        self.assertEqual(hash3, md.refhash)

        composition.filter_min_length(50000000000)
        self.assertEqual(md.nseqs, 0)
        self.assertFalse(np.any(md.mask))

    def test_minlength(self):
        with self.assertRaises(ValueError):
            Composition.from_file(self.io, minlength=3)

    def test_properties(self):
        composition = Composition.from_file(self.io, minlength=420)
        passed = list(filter(lambda x: len(x.sequence) >= 420, self.records))

        self.assertEqual(composition.nseqs, len(composition.metadata.identifiers))
        self.assertEqual(composition.nseqs, len(composition.metadata.lengths))

        self.assertTrue(composition.matrix.dtype, np.float32)
        self.assertEqual(composition.matrix.shape, (len(passed), 103))

        # Names
        self.assertEqual(list(composition.metadata.identifiers), [i.header for i in passed])

        # Lengths
        self.assertTrue(np.issubdtype(composition.metadata.lengths.dtype, np.integer))
        self.assertEqual([len(i.sequence) for i in passed], list(composition.metadata.lengths))

    def test_save_load(self):
        buf = io.BytesIO()
        composition_1 = Composition.from_file(self.io)
        md1 = composition_1.metadata
        composition_1.save(buf)
        buf.seek(0)
        composition_2 = Composition.load(buf)
        md2 = composition_2.metadata
        
        self.assertTrue(np.all(composition_1.matrix == composition_2.matrix))
        self.assertTrue(np.all(md1.identifiers == md2.identifiers))
        self.assertTrue(np.all(md1.lengths == md2.lengths))
        self.assertTrue(np.all(md1.refhash == md2.refhash))
        self.assertTrue(np.all(md1.minlength == md2.minlength))
