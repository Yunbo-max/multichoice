"""Deterministic Sokoban simulator for *training/oracle* environments.

State is immutable; no hidden mutation of sibling search branches. Search code may
clone states freely in local TRAINING games, NOT in official live ARC-AGI-3.
"""
from dataclasses import dataclass
from typing import FrozenSet, Tuple

MOVES = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
ACTION_ORDER = tuple(MOVES)

@dataclass(frozen=True)
class State:
    player: int
    boxes: FrozenSet[int]

@dataclass(frozen=True)
class Board:
    width: int
    height: int
    walls: FrozenSet[int]
    goals: FrozenSet[int]

    @classmethod
    def from_ascii(cls, data):
        if isinstance(data, str):
            rows = data.splitlines()
        else:
            rows = list(data)
        if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
            raise ValueError("nonempty rectangular ASCII map required")
        wall, goals, boxes, player = set(), set(), set(), []
        for r, line in enumerate(rows):
            for c, char in enumerate(line):
                pos = r * len(rows[0]) + c
                if char == "#":
                    wall.add(pos)
                elif char in (" ", "-"):
                    pass
                elif char in (".", "+", "*"):
                    goals.add(pos)
                elif char not in ("@", "$"):
                    raise ValueError(f"unknown map symbol: {char!r}")
                if char in ("@", "+"):
                    player.append(pos)
                if char in ("$", "*"):
                    boxes.add(pos)
        if len(player) != 1 or len(goals) != len(boxes) or not goals:
            raise ValueError("exactly one player and equal positive box/goal counts required")
        return cls(len(rows[0]), len(rows), frozenset(wall), frozenset(goals)), State(
            player[0], frozenset(boxes))

    def _move_index(self, pos, move):
        dr, dc = MOVES[move]
        r, c = divmod(pos, self.width)
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= self.height or nc < 0 or nc >= self.width:
            return None
        return nr * self.width + nc

    def next_state(self, state: State, move: str):
        """Return a successor or None on wall/illegal box push."""
        if move not in MOVES:
            raise ValueError(f"unknown move {move}")
        ahead = self._move_index(state.player, move)
        if ahead is None or ahead in self.walls:
            return None
        boxes = state.boxes
        if ahead in boxes:
            beyond = self._move_index(ahead, move)
            if beyond is None or beyond in self.walls or beyond in boxes:
                return None
            boxes = (boxes - frozenset((ahead,))) | frozenset((beyond,))
        return State(ahead, frozenset(boxes))

    def legal_actions(self, state):
        return [move for move in ACTION_ORDER if self.next_state(state, move) is not None]

    def won(self, state):
        return state.boxes == self.goals

    def render_ascii(self, state):
        lines = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                k = r*self.width+c
                if k in self.walls: token = "#"
                elif k == state.player: token = "+" if k in self.goals else "@"
                elif k in state.boxes: token = "*" if k in self.goals else "$"
                elif k in self.goals: token = "."
                else: token = " "
                row.append(token)
            lines.append("".join(row))
        return "\n".join(lines)

DEMO_MAP = (
    "#######",
    "#@    #",
    "#  $. #",
    "#     #",
    "#######",
)

def canonical_layout_group(board: Board):
    """Group layouts modulo rotations/reflections of walls and goals.

    Player/box locations and language variants of the same game must stay in
    one train or DEV split. Full-game evaluation uses separate official layouts.
    """
    import hashlib
    import json
    candidates=[]
    for rot in range(4):
        hh, ww = ((board.height, board.width) if rot%2==0
                  else (board.width, board.height))
        def rotate(index):
            r,c=divmod(index,board.width)
            if rot==0: return r,c
            if rot==1: return c,board.height-1-r
            if rot==2: return board.height-1-r,board.width-1-c
            return board.width-1-c,r
        for mirror in (False,True):
            def indices(cells):
                values=[]
                for pos in cells:
                    rr,cc=rotate(pos)
                    if mirror: cc=ww-1-cc
                    values.append(rr*ww+cc)
                return tuple(sorted(values))
            candidates.append((hh,ww,indices(board.walls),indices(board.goals)))
    canonical=min(candidates)
    payload=json.dumps(canonical,separators=(",",":"))
    return "layout-"+hashlib.sha256(payload.encode()).hexdigest()[:24]
