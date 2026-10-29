"""CPU-only verification of exact multi-optimal Sokoban supervision."""
import tempfile
import unittest
from pathlib import Path
from treejev.sokoban import ACTION_ORDER, Board, DEMO_MAP
from treejev.oracle import shortest_path_dag, SearchBudgetExceeded
from treejev.data import (
    ensure_action_probs, generate_oracle_level, read_jsonl, save_jsonl,
    split_groups, valen_records)
from treejev.beam import beam_search

class TreeSearchTests(unittest.TestCase):
    def setUp(self):
        self.board,self.initial=Board.from_ascii(DEMO_MAP)

    def test_two_optimal_first_actions_and_trajectories(self):
        dag=shortest_path_dag(self.board,self.initial)
        self.assertEqual(dag.distance,3)
        self.assertEqual(set(dag.optimal[self.initial]),{"right","down"})
        routes=dag.trajectories(limit=20)
        self.assertEqual(len(routes),2)
        for route in routes:
            state=self.initial
            for a in route:
                state=self.board.next_state(state,a)
                self.assertIsNotNone(state)
            self.assertTrue(self.board.won(state))
        self.assertEqual(dag.count_paths(),2)

    def test_budget_never_silently_calls_partial_search_optimal(self):
        with self.assertRaises(SearchBudgetExceeded):
            shortest_path_dag(self.board,self.initial,max_nodes=2)

    def test_immutable_branches(self):
        child1=self.board.next_state(self.initial,"right")
        child2=self.board.next_state(self.initial,"down")
        self.assertNotEqual(child1,child2)
        self.assertEqual(self.board.render_ascii(self.initial),"\n".join(DEMO_MAP))

    def test_generate_rows_and_group_isolation(self):
        rows,trajs,stats=generate_oracle_level("tiny",self.board,self.initial)
        first=next(row for row in rows if row["oracle_depth"]==0)
        self.assertEqual(first["target"],[0,1,1,0])
        self.assertEqual(len(trajs),2)
        self.assertEqual(stats["distance"],3)
        with tempfile.TemporaryDirectory() as temp:
            f=Path(temp)/"items.jsonl"
            save_jsonl(f,rows)
            self.assertEqual(len(list(read_jsonl(f))),len(rows))
            with self.assertRaises(FileExistsError):
                save_jsonl(f,rows)
        copy=[dict(r,group_id="level2",id=r["id"]+"x") for r in rows]
        train,dev=split_groups(rows+copy,dev_fraction=.5)
        self.assertTrue(train and dev)
        self.assertFalse({r["group_id"] for r in train}&{r["group_id"] for r in dev})

    def test_valen_schema_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/"assets").mkdir()
            (root/"assets"/"1.png").write_bytes(b"placeholder")
            sample={"id":"s1","group_id":"board1",
                    "request":{"state":{"messages":["visible"]}},
                    "targets":{"next_action":{"probabilities":{
                       "up":0.0,"right":0.5,"down":0.5,"left":0.0}}},
                    "meta":{"secret":"must never enter model input"},
                    "assets":[{"path":"assets/1.png"}]}
            f=root/"train.jsonl";save_jsonl(f,[sample])
            records=valen_records(f,root)
            self.assertEqual(records[0]["optimal_actions"],["right","down"])
            self.assertNotIn("meta",records[0])
            self.assertNotIn("request",records[0])
            forbidden=root/"eval_sokoban"/"single_step.jsonl"
            with self.assertRaises(ValueError):
                valen_records(forbidden,root,require_images=False)

    def test_simple_beam_recovers_solution(self):
        uniform=lambda state:{a:0.0 for a in self.board.legal_actions(state)}
        result=beam_search(self.board,self.initial,uniform,width=20,horizon=6)
        self.assertTrue(result.won)
        self.assertEqual(result.depth,3)

    def test_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            ensure_action_probs({"up":1.0})
        with self.assertRaises(ValueError):
            ensure_action_probs(dict.fromkeys(ACTION_ORDER,0.0))

if __name__=="__main__":
    unittest.main()
