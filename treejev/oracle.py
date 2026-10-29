"""Exact shortest-path DAG in a bounded deterministic Sokoban training map.

Certified labels are emitted only when BFS finished exploring all predecessor
states for the first goal depth. If a search budget is exceeded, raise instead
of quietly labeling a suboptimal partial search as 'optimal'.
"""
from collections import deque
from dataclasses import dataclass
from treejev.sokoban import ACTION_ORDER, Board, State

class SearchBudgetExceeded(RuntimeError):
    pass

class NoSolutionWithinBudget(RuntimeError):
    pass

@dataclass
class ShortestDAG:
    board: Board
    root: State
    distance: int
    depths: dict
    optimal: dict

    def good_states(self):
        return [s for s in self.depths if s in self.optimal or self.board.won(s)]

    def count_paths(self, cap=10**9):
        counts = {s: 1 for s in self.depths if self.board.won(s)
                  and self.depths[s] == self.distance}
        for depth in range(self.distance-1, -1, -1):
            for state, d in self.depths.items():
                if d == depth and state in self.optimal:
                    counts[state] = min(cap, sum(counts.get(self.board.next_state(state, a), 0)
                                            for a in self.optimal[state]))
        return counts.get(self.root, 0)

    def trajectories(self, limit=32):
        """Enumerate at most limit shortest successful action sequences."""
        if limit < 1:
            raise ValueError("limit must be positive")
        found = []
        def visit(state, path):
            if len(found) >= limit:
                return
            if self.board.won(state):
                found.append(list(path))
                return
            for a in self.optimal.get(state, ()):
                nxt = self.board.next_state(state, a)
                visit(nxt, path + (a,))
                if len(found) >= limit:
                    break
        visit(self.root, ())
        return found

def shortest_path_dag(board, initial, *, max_nodes=20000, max_depth=80):
    if max_nodes < 1 or max_depth < 0:
        raise ValueError("invalid search budget")
    queue = deque([initial]); depths = {initial: 0}; edges = {}
    best = None
    while queue:
        state = queue.popleft()
        d = depths[state]
        if board.won(state):
            if best is None: best = d
            continue
        if best is not None and d >= best:
            continue
        if d >= max_depth:
            continue
        current = []
        for action in board.legal_actions(state):
            nxt = board.next_state(state, action)
            if nxt not in depths:
                if len(depths) >= max_nodes:
                    raise SearchBudgetExceeded("search node budget exhausted; no exact label produced")
                depths[nxt] = d+1
                queue.append(nxt)
            if depths[nxt] == d+1:
                current.append((action, nxt))
        edges[state] = current
    if best is None:
        raise NoSolutionWithinBudget("no solution proven within bounded search; NOT a proof of unsolvability")
    winners = {s for s, d in depths.items() if d == best and board.won(s)}
    good = set(winners); optimal = {}
    for d in range(best-1, -1, -1):
        for state, depth in depths.items():
            if depth != d:
                continue
            actions = tuple(a for a, nxt in edges.get(state, ()) if nxt in good)
            if actions:
                optimal[state] = actions
                good.add(state)
    if initial not in good:
        raise AssertionError("BFS found goal but shortest-path backtrace missed root")
    return ShortestDAG(board, initial, best, depths, optimal)
