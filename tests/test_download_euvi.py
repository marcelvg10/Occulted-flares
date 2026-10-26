"""Offline acquisition tests; actual archive connectivity is a separate check."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from occulted_flares.geometry.download_euvi import read_config, run
from occulted_flares.geometry.geometry_runtime import configure_cache
configure_cache()
from astropy.table import QTable
from sunpy.net import Fido


class DownloadTest(unittest.TestCase):
    def config(self, root):
        path = root / "event.json"
        path.write_text(json.dumps(dict(start_utc="2026-07-07T22:40:00", end_utc="2026-07-07T23:10:00",
                                       source="STEREO_A", wavelengths_angstrom=[195, 304],
                                       destination="fits", reports="reports", max_files=10)))
        return path

    def manifest(self, root):
        return json.loads(next((root/"reports").glob("*/manifest.json")).read_text())

    def test_config_paths_and_invalid_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.config(root)
            self.assertEqual(read_config(path)["destination"], str(root/"fits"))
            data = json.loads(path.read_text())
            data["end_utc"] = "2026-07-07T21:00:00"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "end_utc"):
                read_config(path)

    def test_no_matches_does_not_download(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(Fido, "search", return_value=[]), patch.object(Fido, "fetch") as fetch:
                self.assertEqual(run(self.config(root), True), 2)
                fetch.assert_not_called()
            self.assertEqual(self.manifest(root)["status"], "no_matches")

    def test_search_table_drops_live_client_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            table = QTable({"fileid": ["test-observation"]})
            table.meta["client"] = object()  # VSO objects cannot be serialized to ECSV.
            with patch.object(Fido, "search", return_value=[table]), patch.object(Fido, "fetch") as fetch:
                self.assertEqual(run(self.config(root), False), 0)
                fetch.assert_not_called()
            self.assertEqual(self.manifest(root)["match_count"], 2)
            self.assertEqual(len(list((root/"reports").glob("*/*.ecsv"))), 2)


if __name__ == "__main__":
    unittest.main()
