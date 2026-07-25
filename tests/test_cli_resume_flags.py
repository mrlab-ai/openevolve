"""
Test CLI resume flags: --auto-resume and --until-iteration
"""

import io
import os
import tempfile
import unittest
from unittest.mock import patch

from openevolve.cli import find_latest_checkpoint, parse_args


class TestFindLatestCheckpoint(unittest.TestCase):
    """Test discovery of the newest checkpoint directory"""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.checkpoints = os.path.join(self.tmpdir.name, "checkpoints")
        os.makedirs(self.checkpoints)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _make(self, name):
        path = os.path.join(self.checkpoints, name)
        os.makedirs(path)
        return path

    def test_missing_directory(self):
        """Missing checkpoints directory yields None"""
        self.assertIsNone(find_latest_checkpoint(os.path.join(self.tmpdir.name, "nope")))

    def test_empty_directory(self):
        """Empty checkpoints directory yields None"""
        self.assertIsNone(find_latest_checkpoint(self.checkpoints))

    def test_picks_numerically_highest(self):
        """Checkpoints are ordered numerically, not lexicographically"""
        self._make("checkpoint_5")
        self._make("checkpoint_50")
        expected = self._make("checkpoint_120")

        self.assertEqual(find_latest_checkpoint(self.checkpoints), expected)

    def test_ignores_unrelated_entries(self):
        """Non-checkpoint directories and files are skipped"""
        expected = self._make("checkpoint_2")
        self._make("not_a_checkpoint")
        self._make("checkpoint_abc")
        with open(os.path.join(self.checkpoints, "checkpoint_999"), "w") as f:
            f.write("a file, not a directory")

        self.assertEqual(find_latest_checkpoint(self.checkpoints), expected)


class TestResumeFlagValidation(unittest.TestCase):
    """Test argument validation for the resume flags"""

    def _parse(self, *extra):
        argv = ["openevolve-run.py", "prog.py", "eval.py", *extra]
        # argparse writes usage text to stderr before exiting on invalid input
        with patch("sys.argv", argv), patch("sys.stderr", io.StringIO()):
            return parse_args()

    def test_defaults(self):
        """Resume flags default to off"""
        args = self._parse()
        self.assertFalse(args.auto_resume)
        self.assertIsNone(args.until_iteration)

    def test_flags_parsed(self):
        """Both flags and their short forms are accepted"""
        args = self._parse("-r", "-u", "500")
        self.assertTrue(args.auto_resume)
        self.assertEqual(args.until_iteration, 500)

    def test_auto_resume_conflicts_with_checkpoint(self):
        """--auto-resume and --checkpoint are mutually exclusive"""
        with self.assertRaises(SystemExit):
            self._parse("--auto-resume", "--checkpoint", "some/checkpoint_10")

    def test_until_iteration_conflicts_with_iterations(self):
        """--until-iteration and --iterations are mutually exclusive"""
        with self.assertRaises(SystemExit):
            self._parse("--until-iteration", "100", "--iterations", "10")

    def test_until_iteration_must_be_positive(self):
        """--until-iteration below 1 is rejected"""
        for value in ("0", "-5"):
            with self.subTest(value=value):
                with self.assertRaises(SystemExit):
                    self._parse("--until-iteration", value)


if __name__ == "__main__":
    unittest.main()
