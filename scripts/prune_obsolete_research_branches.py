#!/usr/bin/env python3
"""Safely prune precisely nine audited intermediate research refs.

DRY RUN by default; requires --delete and typed interactive confirmation for
real remote ref deletions. GitHub app in this chat does not expose delete-ref.

Run inside an authenticated clone of Yunbo-max/multichoice:
  python scripts/prune_obsolete_research_branches.py
  python scripts/prune_obsolete_research_branches.py --delete

Refuses if any old head SHA changed or is not reachable from the retained tip.
Every old commit remains in retained branch ancestry after a branch deletion.
"""
import argparse
import re
import subprocess
import sys

# Exact audit snapshot, 2026-11-18.
KEEP = (
    "main",
    "research/static-setjev-paper-20261027",
    "research/romqa-anchor-icml2027-20261114",
    "research/episodic-tree-sokoban-20261027",
)
OLD = (
    ("codex/multiteacher-set-distillation",
     "c77214b5c930140f4767cffea6deee83cf2f842f", KEEP[2]),
    ("research/recurrent-set-decision-20261024",
     "07d152b12018e3cc92deea577e80e18af255c82c", KEEP[3]),
    ("research/recurrent-set-debug-20261024",
     "735bc69b83414e12ba80407d2ddd0865a17ca2f2", KEEP[3]),
    ("research/episodic-set-memory-20261024",
     "823ec93ca6ed3366a7f2ab71ec125031e7a80a3a", KEEP[3]),
    ("research/episodic-set-debug-20261025",
     "c34ef0982f5eecf8c02400a18813a9d76f46d6f7", KEEP[3]),
    ("research/episodic-identifiability-debug-20261025",
     "aa793886851dc46c2b542f03c3a4a39e966c02cd", KEEP[3]),
    ("research/episodic-probe-group-split-20261025",
     "d8e56f03c302d248cd625766f61290b706a92757", KEEP[3]),
    ("research/episodic-action-source-diagnostic-20261025",
     "8fcbc13bf0156f9e8eb19406275ce42f40798735", KEEP[3]),
    ("research/episodic-lora-reliability-20261026",
     "e3923206fa59c79d720fe9cabf569eaabcdfc4a3", KEEP[3]),
)


def git(*args, check=True):
    return subprocess.run(["git", *args], text=True, capture_output=True,
                          check=check)


def check_all():
    root = git("rev-parse", "--show-toplevel").stdout.strip()
    origin = git("remote", "get-url", "origin").stdout.strip()
    trusted = (
        r"https://github\.com/Yunbo-max/multichoice(?:\.git)?",
        r"git@github\.com:Yunbo-max/multichoice(?:\.git)?",
        r"ssh://git@github\.com/Yunbo-max/multichoice(?:\.git)?",
    )
    if not any(re.fullmatch(pattern, origin) for pattern in trusted):
        raise RuntimeError("Refusing to act on an unexpected origin: " + origin)
    if any(branch in KEEP for branch, _, _ in OLD):
        raise RuntimeError("Attempt to remove a protected retained branch")
    git("fetch", "origin")
    for survivor in KEEP:
        if git("rev-parse", "--verify", "refs/remotes/origin/" + survivor,
               check=False).returncode:
            raise RuntimeError("Preserved branch absent: " + survivor)
    for old, expected, survivor in OLD:
        result = git("rev-parse", "--verify", "refs/remotes/origin/" + old,
                     check=False)
        if result.returncode:
            raise RuntimeError("Missing old ref (may already be gone): " + old)
        if result.stdout.strip() != expected:
            raise RuntimeError("ABORT! Old branch moved since audit: " + old)
        ancestry = git("merge-base", "--is-ancestor", expected,
                       "refs/remotes/origin/" + survivor, check=False)
        if ancestry.returncode:
            raise RuntimeError("ABORT! Old branch is not in preserved ancestry: " + old)
    return root


def remote_heads():
    lines = git("ls-remote", "--heads", "origin").stdout.splitlines()
    return {row.rsplit("refs/heads/", 1)[-1] for row in lines
            if "refs/heads/" in row}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete after SHA/ancestry checks and typed confirmation")
    args = parser.parse_args(argv)
    root = check_all()
    print("Verified origin for checkout:", root)
    print("Preserved refs:", ", ".join(KEEP))
    print("Exact audited old ref candidates:")
    for old, sha, parent in OLD:
        print("  -", old, sha[:12], "contained in", parent)
    if not args.delete:
        print("DRY RUN. No refs deleted. Use --delete only after reviewing.")
        return 0
    if input("Type DELETE 9 BRANCHES to confirm: ") != "DELETE 9 BRANCHES":
        print("No changes.")
        return 1
    for old, sha, parent in OLD:
        print("Deleting origin/" + old)
        try:
            git("push", "origin", "--delete", old)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("Deletion stopped at " + old + ": " +
                               (exc.stderr or str(exc))) from exc
    remaining = remote_heads()
    if any(name in remaining for name, _, _ in OLD):
        raise RuntimeError("Not all obsolete remote refs were removed")
    if any(name not in remaining for name in KEEP):
        raise RuntimeError("A retained branch is unexpectedly missing")
    print("Verified 9 obsolete refs removed; all 4 retained refs still live.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print("ERROR:", str(error), file=sys.stderr)
        sys.exit(1)
