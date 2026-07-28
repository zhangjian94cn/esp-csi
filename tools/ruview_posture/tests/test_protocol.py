from __future__ import annotations

import unittest

from ..protocol import (
    CSI_HEADER,
    DISCOVERY,
    STATUS,
    CsiPacket,
    ProtocolError,
    StatusPacket,
    decode_packet,
    encode_discovery,
)
from .fixtures import csi_datagram, status_datagram


class ProtocolTest(unittest.TestCase):
    def test_struct_sizes_match_firmware_contract(self) -> None:
        self.assertEqual(CSI_HEADER.size, 48)
        self.assertEqual(STATUS.size, 64)
        self.assertEqual(DISCOVERY.size, 16)

    def test_decodes_csi_golden_fixture(self) -> None:
        packet = decode_packet(
            csi_datagram(node_id=2, sequence=17, csi=b"\x01\x02" * 32)
        )
        self.assertIsInstance(packet, CsiPacket)
        self.assertEqual(packet.node_id, 2)
        self.assertEqual(packet.sequence, 17)
        self.assertEqual(packet.tx_mac, "28:84:85:92:81:3c")
        self.assertEqual(packet.gain_compensation, 1.0)
        self.assertEqual(len(packet.csi), 64)

    def test_decodes_status_golden_fixture(self) -> None:
        packet = decode_packet(status_datagram())
        self.assertIsInstance(packet, StatusPacket)
        self.assertEqual(packet.frames_sent, 245)
        self.assertEqual(packet.build_id, "8633d671-rvp1")

    def test_rejects_length_mismatch(self) -> None:
        with self.assertRaises(ProtocolError):
            decode_packet(
                csi_datagram(node_id=2, sequence=1, csi=b"\x01\x02" * 32)[:-1]
            )

    def test_discovery_is_versioned(self) -> None:
        packet = encode_discovery(sink_port=5006, nonce=123)
        magic, version, _, port, nonce, ttl = DISCOVERY.unpack(packet)
        self.assertEqual(version, 1)
        self.assertEqual(port, 5006)
        self.assertEqual(nonce, 123)
        self.assertEqual(ttl, 10_000)


if __name__ == "__main__":
    unittest.main()
