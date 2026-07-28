from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

from ..validate_backup import FLASH_SIZE, PARTITION_ENTRY, validate


class BackupValidationTest(unittest.TestCase):
    def test_parses_valid_16mb_partition_table(self) -> None:
        image = bytearray(b"\xff" * FLASH_SIZE)
        entries = [
            (0x50AA, 1, 2, 0x9000, 0x6000, b"nvs", 0),
            (0x50AA, 0, 0, 0x10000, 0x180000, b"factory", 0),
        ]
        for index, entry in enumerate(entries):
            magic, type_, subtype, offset, size, label, flags = entry
            label = label.ljust(16, b"\0")
            start = 0x8000 + index * PARTITION_ENTRY.size
            image[start : start + PARTITION_ENTRY.size] = PARTITION_ENTRY.pack(
                magic, type_, subtype, offset, size, label, flags
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backup.bin"
            path.write_bytes(image)
            result = validate(path)
        self.assertTrue(result["valid"], result["failures"])
        self.assertEqual(result["partition_table_offset"], 0x8000)
        self.assertEqual(len(result["partitions"]), 2)


if __name__ == "__main__":
    unittest.main()
