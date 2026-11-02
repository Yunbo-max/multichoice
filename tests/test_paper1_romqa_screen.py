"""No-download tests for the official RoMQA closed-format evidence/count audit."""
from __future__ import annotations
import bz2
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from paper_static.romqa_data_screen import screen,load_zip,load_dir,main


def case(id_,cluster,positive,*,k=4,constraints=2):
    return {"id":id_,"cluster_id":cluster,
            "question":f"Which entities satisfy {constraints} constraints?",
            "constraints":[f"constraint {i}" for i in range(constraints)],
            "candidates":[
                {"text":f"entity {id_}-{j}", "is_answer":j in positive,
                 "dpr":["retrieved snippet"],"dpr_score":[.68]}
                for j in range(k)],
            "complete_answer":[{"text":f"entity {id_}-{j}"}
                               for j in positive]}


class RoMQAAudit(unittest.TestCase):
    def setUp(self):
        self.train=[case("tr-0","contrast-A",[0]),
                    case("tr-1","contrast-A",[0,1,2]),
                    case("tr-2","contrast-B",[1,2])]
        self.dev=[case("dv-0","contrast-C",[0,2]),
                  case("dv-1","contrast-C",[])]
    def test_count_histogram_contrast_clusters_and_gold_coverage(self):
        out=screen({"train":self.train,"dev":self.dev})
        self.assertFalse(out["test_opened"])
        self.assertEqual(out["splits"]["train"]["questions"],3)
        self.assertEqual(out["splits"]["train"]["clusters_with_varying_answer_count"],1)
        self.assertEqual(out["splits"]["train"]["positive_count_histogram"],{1:1,2:1,3:1})
        self.assertEqual(out["splits"]["dev"]["positive_count_histogram"],{0:1,2:1})
        self.assertEqual(out["splits"]["dev"]["empty_answer_fraction"],.5)
        self.assertFalse(out["decision_gate"]["count_gap_measured"])
        self.assertIn("not every question",out["issues"][0])

    def test_zip_matches_directory_and_only_train_dev_opened(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            closed=root/"data"/"closed"
            closed.mkdir(parents=True)
            for name,rows in (("train",self.train),("dev",self.dev)):
                (closed/f"top_20.{name}.json.bz2").write_bytes(
                    bz2.compress(json.dumps(rows).encode()))
            archive=root/"romqa_data.zip"
            with zipfile.ZipFile(archive,"w") as z:
                for file in closed.glob("*.bz2"):
                    z.write(file,file.relative_to(root))
                z.writestr("data/closed/top_20.test.noanswer.json.bz2",b"test-hidden-corrupt")
            self.assertEqual(load_zip(archive),load_dir(closed))
            output=root/"audit.json"
            main(["--archive",str(archive),"--output",str(output)])
            got=json.loads(output.read_text())
            self.assertFalse(got["test_opened"])
            self.assertEqual(got["source_kind"],"third_party_zip")
            self.assertEqual(got["splits"]["dev"]["questions"],2)
            with self.assertRaises(SystemExit):
                main(["--archive",str(archive),"--output",str(output)])

    def test_reject_cross_split_cluster_leakage(self):
        with self.assertRaisesRegex(ValueError,"leaks across"):
            screen({"train":self.train,
                    "dev":[case("new","contrast-A",[1])]})

    def test_reject_unlabeled_dev_and_hidden_test(self):
        invalid=case("bad","contrast-C",[0])
        invalid["candidates"][0].pop("is_answer")
        with self.assertRaisesRegex(ValueError,"unlabeled candidate"):
            screen({"train":self.train,"dev":[invalid]})
        with self.assertRaisesRegex(ValueError,"ONLY train and dev"):
            screen({"train":self.train,"dev":self.dev,"test":[]})

    def test_complete_answer_outside_candidates_quantified(self):
        item=case("dv-extra","other",[0,1])
        item["complete_answer"].append({"text":"entity outside candidate pool"})
        out=screen({"train":self.train,"dev":[item]})
        self.assertAlmostEqual(
            out["splits"]["dev"]["mean_fraction_complete_answers_outside_candidates"],1/3)


if __name__=="__main__":
    unittest.main()
