"""Exact remote mutation evidence never implies source/readiness authorization."""
import unittest

from app.voice_routes.contracts import ContextWriteEvidence


class ContextCreateAckEvidenceTests(unittest.TestCase):
    def test_exact_owned_ack_requires_closed_joined_attempt_and_stays_written(self):
        proof = ContextWriteEvidence()
        proof.begin_tracking()
        self.assertTrue(proof.before_write("context_create", item_id="owned"))
        self.assertFalse(proof.record_create_acknowledged("foreign"))
        self.assertFalse(proof.created_context_acknowledged)
        self.assertTrue(proof.record_create_acknowledged("owned"))
        self.assertTrue(proof.record_create_acknowledged("owned"))
        self.assertTrue(proof.created_context_acknowledged)
        self.assertFalse(proof.confirmed_created)
        proof.close()
        self.assertFalse(proof.confirmed_created)
        proof.mark_joined()
        self.assertTrue(proof.confirmed_created)
        self.assertEqual(proof.state, "write_possible")
        self.assertFalse(proof.proven_unwritten)
        self.assertFalse(proof.confirmed_deleted_without_create)

    def test_untracked_missing_foreign_multiple_late_and_unjoined_are_unknown(self):
        for fault in ("untracked", "missing_id", "foreign", "multiple", "late", "unjoined"):
            with self.subTest(fault=fault):
                proof = ContextWriteEvidence()
                if fault != "untracked":
                    proof.begin_tracking()
                proof.before_write("context_create", item_id=None if fault == "missing_id" else "owned")
                if fault == "multiple":
                    proof.before_write("context_create", item_id="owned")
                if fault == "late":
                    proof.close()
                proof.record_create_acknowledged("foreign" if fault == "foreign" else "owned")
                proof.close()
                if fault != "unjoined":
                    proof.mark_joined()
                self.assertFalse(proof.confirmed_created)
                self.assertFalse(proof.proven_unwritten)

    def test_replacement_requires_exact_preceding_delete_ack(self):
        for matched in (False, True):
            with self.subTest(matched=matched):
                proof = ContextWriteEvidence()
                proof.begin_tracking()
                proof.before_write("context_delete", item_id="old")
                proof.record_delete_acknowledged("old" if matched else "foreign")
                proof.before_write("context_create", item_id="new")
                self.assertTrue(proof.record_create_acknowledged("new"))
                proof.close()
                proof.mark_joined()
                self.assertEqual(proof.confirmed_created, matched)

    def test_following_delete_does_not_retain_created_only_proof(self):
        proof = ContextWriteEvidence()
        proof.begin_tracking()
        proof.before_write("context_create", item_id="new")
        proof.record_create_acknowledged("new")
        proof.before_write("context_delete", item_id="new")
        self.assertFalse(proof.record_delete_acknowledged("new"))
        proof.close()
        proof.mark_joined()
        self.assertFalse(proof.confirmed_created)


if __name__ == "__main__":
    unittest.main()
