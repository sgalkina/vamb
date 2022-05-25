import vamb
import re
import subprocess
import unittest


def grep(path, regex):
    with open(path) as file:
        for line in file:
            m = regex.search(line)
            if m is not None:
                g = m.groups()
                return (int(g[0]), int(g[1]), int(g[2]))

    raise ValueError(f"Could not find regex in path {path}")


def snakemake_vamb_version(path):
    regex = re.compile(
        r"github\.com/RasmussenLab/vamb@v([0-9]+)\.([0-9]+)\.([0-9]+)\s*$")
    return grep(path, regex)


def setup_vamb_version(path):
    regex = re.compile(
        r"^\s*\"version\": \"([0-9]+)\.([0-9]+)\.([0-9]+)\"(,\s*)?$")
    return grep(path, regex)


def validate_init(init):
    if not (
        isinstance(init, tuple) and
        len(init) in (3, 4) and
        all(isinstance(i, int) for i in init[:3]) and
        (len(init) == 3 or init[3] == "DEV")
    ):
        raise ValueError("Wrong format of __version__ in __init__.py")


def latest_git_tag():
    st = subprocess.run(["git", "describe", "--tags",
                        "--abbrev=0"], capture_output=True).stdout.decode()
    regex = re.compile(r"^v?([0-9]+)\.([0-9]+)\.([0-9])\n?$")
    m = regex.match(st)
    if m is None:
        raise ValueError("Could not find last git tag")
    else:
        return tuple(int(i) for i in m.groups())


def head_git_tag():
    st = subprocess.run(["git", "tag", "--points-at", "HEAD"],
                        capture_output=True).stdout.decode()
    if len(st) == 0:
        return (None, None)
    regex = re.compile(r"^(v([0-9]+)\.([0-9]+)\.([0-9]))\n?$")
    m = regex.match(st)
    if m is None:
        raise ValueError("HEADs git tag is not a valid version number")
    vnum = tuple(int(i) for i in m.groups()[1:4])
    tagname = m.groups()[0]

    # Check it's annotated if it exists - then returncode is 0
    proc = subprocess.run(["git", "describe", tagname])
    is_annotated = proc.returncode == 0
    return (vnum, is_annotated)


class TestVersions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v_setup = setup_vamb_version("../setup.py")
        cls.v_snakemake = snakemake_vamb_version("../workflow/envs/vamb.yaml")
        validate_init(vamb.__version__)
        cls.v_init = vamb.__version__
        cls.last_tag = latest_git_tag()
        head_tag, is_annotated = head_git_tag()
        cls.head_tag = head_tag
        cls.is_annotated = is_annotated

    def test_same_versions(self):
        # setup.py, envs/vamb version and last tag must all point to the latest release
        self.assertEqual(self.v_setup, self.v_snakemake)
        self.assertEqual(self.v_setup, self.last_tag)

    def test_dev_version(self):
        # If the current version is a DEV version, it must be a greater version
        # than the latest release.
        # If not, it must be the same version as the tag of the current commit,
        # i.e. the current commit must be a release version.
        if self.v_init[-1] == "DEV":
            self.assertGreater(self.v_init[:3], self.v_setup)
        else:
            self.assertEqual(self.v_init, self.head_tag)
            self.assertTrue(self.is_annotated)
