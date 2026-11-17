"""Guard-only synthetic regressions. Never call evaluate, fit or real inference."""
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import d04_pipeline as q
import verify_d04 as v


def session_worker(root, run, gate, release, results):
    """A real OS process, but only synthetic registration and an empty body."""
    try:
        with patch.object(q, "ROOT", Path(root)), patch.object(q, "check_declaration", return_value={"config": {"model_version": "synthetic"}}):
            if not gate.wait(15):
                raise RuntimeError("Test start timed out")
            with q.evaluation_session(Path(run)):
                results.put(("acquired", str(run)))
                if not release.wait(15):
                    raise RuntimeError("Test release timed out")
    except ValueError as exc:
        results.put(("rejected", str(exc)))
    except BaseException as exc:
        results.put(("error", repr(exc)))


class CheckpointCommitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="d04-commit-")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        (self.run / "checkpoints").mkdir()
        (self.run / "events.jsonl").write_text("")

    def file(self, kind="features", offset=0):
        path = self.run / f"checkpoints/{offset:03d}_{kind}.npz"
        path.write_bytes(b"synthetic checkpoint; digest validation needs no array decoding")
        return path

    def commit(self, path, event="features_saved", offset=0, digest=None):
        q.event(self.run, event, offset=offset, sha256=digest or q.p.sha(path))

    def checks(self):
        events = [json.loads(line) for line in (self.run / "events.jsonl").read_text().splitlines()]
        return (("recovery", lambda: q.verify_checkpoint_log(self.run)),
                ("independent", lambda: v.check_checkpoint_commits(self.run, events)))

    def refuse(self, message):
        for name, check in self.checks():
            with self.subTest(checker=name), self.assertRaisesRegex(ValueError, message):
                check()

    def test_empty_before_first_batch(self):
        for _, check in self.checks():
            self.assertFalse(check())

    def test_feature_renamed_before_commit_is_rejected(self):
        self.file()
        self.refuse("files and original commits differ")

    def test_prediction_renamed_before_commit_is_rejected(self):
        self.commit(self.file())
        self.file("predictions")
        self.refuse("files and original commits differ")

    def test_reuse_cannot_bless_uncommitted_file(self):
        self.commit(self.file(), "feature_checkpoint_reused")
        self.refuse("reuse without original commit")

    def test_reuse_before_original_commit_rejected(self):
        path = self.file()
        self.commit(path, "feature_checkpoint_reused")
        self.commit(path)
        self.refuse("reuse without original commit")

    def test_duplicate_original_commit_even_same_hash_rejected(self):
        path = self.file(); self.commit(path); self.commit(path)
        self.refuse("Duplicate checkpoint commit")

    def test_missing_committed_file_rejected(self):
        path = self.file(); self.commit(path); path.unlink()
        self.refuse("files and original commits differ")

    def test_changed_committed_bytes_rejected(self):
        path = self.file(); self.commit(path); path.write_bytes(b"changed")
        self.refuse("Checkpoint changed")

    def test_conflicting_reuse_digest_rejected(self):
        path = self.file(); self.commit(path)
        self.commit(path, "feature_checkpoint_reused", digest="0" * 64)
        self.refuse("reuse digest differs")

    def test_prediction_commit_needs_feature_commit(self):
        self.commit(self.file("predictions"), "predictions_saved")
        self.refuse("Prediction commit without feature commit")

    def test_unexpected_file_rejected(self):
        self.commit(self.file()); (self.run / "checkpoints/unknown.npz").write_bytes(b"orphan")
        self.refuse("files and original commits differ")

    def test_valid_original_commits_and_multiple_reuses(self):
        fp = self.file(); pp = self.file("predictions")
        self.commit(fp); self.commit(pp, "predictions_saved")
        for _ in range(2):
            self.commit(fp, "feature_checkpoint_reused")
            self.commit(pp, "prediction_checkpoint_reused")
        for _, check in self.checks():
            self.assertEqual(len(check()), 2)


class SessionLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="d04-lock-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.a = self.root / "runs/d04/a"
        self.b = self.root / "runs/d04/b"
        for run in (self.a, self.b):
            run.mkdir(parents=True)
            (run / "declaration.json").write_text('{"synthetic":true}')
        self.root_patch = patch.object(q, "ROOT", self.root)
        self.root_patch.start(); self.addCleanup(self.root_patch.stop)
        self.declaration_patch = patch.object(q, "check_declaration", return_value={"config": {"model_version": "synthetic"}})
        self.declaration = self.declaration_patch.start(); self.addCleanup(self.declaration_patch.stop)
        self.ctx = multiprocessing.get_context("spawn")
        self.children = []
        self.addCleanup(self.cleanup_children)

    def cleanup_children(self):
        for child in self.children:
            if child.is_alive(): child.terminate()
            child.join(timeout=5)

    def worker(self, run, gate, release, results):
        child = self.ctx.Process(target=session_worker, args=(str(self.root), str(run), gate, release, results))
        child.start(); self.children.append(child)
        return child

    def test_two_processes_different_runs_only_one_enters(self):
        gate, release, results = self.ctx.Event(), self.ctx.Event(), self.ctx.Queue()
        self.worker(self.a, gate, release, results); self.worker(self.b, gate, release, results)
        gate.set()
        answers = [results.get(timeout=20), results.get(timeout=20)]
        self.assertEqual(sorted(a[0] for a in answers), ["acquired", "rejected"])
        self.assertIn("active", next(a[1] for a in answers if a[0] == "rejected"))
        self.assertEqual(len(list((self.root / "runs/d04").glob("*/execution_started.json"))), 1)
        release.set()
        for child in self.children:
            child.join(timeout=10); self.assertEqual(child.exitcode, 0)
        # Once the OS lock is released, the persisted use record still blocks the other directory.
        loser = self.b if (self.a / "execution_started.json").exists() else self.a
        with self.assertRaisesRegex(ValueError, "Another formal test exists"):
            with q.evaluation_session(loser): self.fail("entered")

    def test_active_process_blocks_same_run_resume_before_declaration(self):
        gate, release, results = self.ctx.Event(), self.ctx.Event(), self.ctx.Queue()
        child = self.worker(self.a, gate, release, results); gate.set()
        self.assertEqual(results.get(timeout=20)[0], "acquired")
        with self.assertRaisesRegex(ValueError, "active"):
            with q.evaluation_session(self.a, resume=True): self.fail("entered")
        self.declaration.assert_not_called()
        release.set(); child.join(timeout=10); self.assertEqual(child.exitcode, 0)

    def test_process_exit_releases_lock_but_preserves_consumed_marker(self):
        gate, release, results = self.ctx.Event(), self.ctx.Event(), self.ctx.Queue()
        child = self.worker(self.a, gate, release, results); gate.set()
        self.assertEqual(results.get(timeout=20)[0], "acquired")
        child.terminate(); child.join(timeout=10)
        with self.assertRaisesRegex(ValueError, "Interrupted evaluation exists"):
            with q.evaluation_session(self.a): self.fail("entered")
        with self.assertRaisesRegex(ValueError, "Another formal test exists"):
            with q.evaluation_session(self.b): self.fail("entered")
        with q.evaluation_session(self.a, resume=True): pass  # registration only, no inference/recovery loop
        events = [json.loads(s)["event"] for s in (self.a / "events.jsonl").read_text().splitlines()]
        self.assertEqual(events, ["formal_test_started", "resume_requested"])

    def test_exception_releases_lock_and_keeps_inode(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with q.evaluation_session(self.a): raise RuntimeError("synthetic failure")
        lock = self.root / "runs/d04/.evaluation.lock"
        inode = lock.stat().st_ino
        with q.evaluation_session(self.a, resume=True): pass
        self.assertEqual(lock.stat().st_ino, inode)

    def test_completed_rejected_before_declaration(self):
        (self.a / "completion.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "Evaluation complete"):
            with q.evaluation_session(self.a): self.fail("entered")
        self.declaration.assert_not_called()

    def test_resume_without_registration_rejected(self):
        with self.assertRaisesRegex(ValueError, "Nothing to resume"):
            with q.evaluation_session(self.a, resume=True): self.fail("entered")
        self.declaration.assert_not_called()

    def test_resume_rejects_changed_declaration(self):
        with q.evaluation_session(self.a): pass
        (self.a / "declaration.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "Recovery declaration changed"):
            with q.evaluation_session(self.a, resume=True): self.fail("entered")


class ReadOnlyCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="d04-readonly-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "runs/d04/original"
        self.run.mkdir(parents=True)
        self.name = "scripts/d04_pipeline.py"
        for base in (self.run / "source", self.root / "before", self.root):
            path = base / self.name; path.parent.mkdir(parents=True); path.write_text("old guards")
        self.declared = {"bound_hashes": {self.name: v.sha(self.root / self.name)}}
        (self.run / "declaration.json").write_text(json.dumps(self.declared))
        (self.run / "completion.json").write_text('{"fixed_result":true}')
        (self.root / self.name).write_text("new guards")
        self.registration = {"purpose": "D04_guard_only_readonly_revision", "run": "runs/d04/original",
             "declaration_sha256": v.sha(self.run / "declaration.json"), "completion_sha256": v.sha(self.run / "completion.json"),
             "changes": {self.name: {"before_sha256": self.declared["bound_hashes"][self.name],
                 "after_sha256": v.sha(self.root / self.name), "before_copy": "before/" + self.name}}}
        self.path = self.root / "compatibility.json"
        self.save()
        self.root_patch = patch.object(v, "ROOT", self.root); self.root_patch.start(); self.addCleanup(self.root_patch.stop)

    def save(self): self.path.write_text(json.dumps(self.registration))
    def check(self): v.check_bindings(self.run, self.declared, self.path)

    def test_explicit_exact_guard_revision_accepted(self): self.check()

    def test_changed_code_without_registration_rejected(self):
        with self.assertRaisesRegex(ValueError, "Bound input/code changed"):
            v.check_bindings(self.run, self.declared)

    def test_registration_cannot_authorize_result_change(self):
        self.registration["changes"]["runs/d04/original/test_predictions.csv"] = {}
        self.save()
        with self.assertRaisesRegex(ValueError, "only change D04 guards"):
            self.check()

    def test_registration_targets_exact_original_completion(self):
        (self.run / "completion.json").write_text('{"different":true}')
        with self.assertRaisesRegex(ValueError, "another evaluation"):
            self.check()

    def test_source_snapshot_cannot_be_replaced(self):
        (self.run / "source" / self.name).write_text("changed source")
        with self.assertRaisesRegex(ValueError, "original source differs"):
            self.check()

    def test_current_guard_change_after_registration_rejected(self):
        (self.root / self.name).write_text("unregistered guards")
        with self.assertRaisesRegex(ValueError, "Unapproved D04 revision"):
            self.check()


if __name__ == "__main__":
    unittest.main()
