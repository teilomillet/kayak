from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import kayak

from kayak_bridge.trec_run import write_trec_run


class TrecRunTests(unittest.TestCase):
    def test_write_trec_run_emits_ranked_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.trec"
            write_trec_run(
                path,
                query_ids=("q1", "q2"),
                hits_by_query=(
                    (
                        kayak.SearchHit(doc_id="doc-a", score=2.0),
                        kayak.SearchHit(doc_id="doc-b", score=1.0),
                    ),
                    (kayak.SearchHit(doc_id="doc-c", score=3.5),),
                ),
                run_name="kayak_test",
            )

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                (
                    "q1 Q0 doc-a 1 2.00000000 kayak_test\n"
                    "q1 Q0 doc-b 2 1.00000000 kayak_test\n"
                    "q2 Q0 doc-c 1 3.50000000 kayak_test\n"
                ),
            )


if __name__ == "__main__":
    unittest.main()
