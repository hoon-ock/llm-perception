# Interpretation notes

Prose arguments about what the results *mean*, each number traced to a committed
`fc_group/Analysis/*/data/*.csv`. Raw material for the manuscript's narrative — the place
where a reading is worked out and its caveats written down before any of it reaches a draft.

**Nothing here is generated.** No script writes these files and no script reads them:
`make_tables.py`, `check_claims.py` and `check_numbers.py` all read the `data/` trees
directly, and must keep doing so. A note here going stale is a note, not a broken pipeline —
but it also means nothing in this directory has been mechanically checked. Numbers are quoted
with their source file named so any of them can be re-read in one command.

Sibling directories (`probe/`, `generation/`, `geometry/`, `ambiguity/`, `taxonomy/`) hold the
analysis scripts and their outputs. Those are the authority. These notes only argue about them.

## Notes

- [`representation_behaviour_gap.md`](representation_behaviour_gap.md) — why the
  teacher-forced ↔ free-generation gap in Table 1 is a *commitment* gap rather than a decoding
  gap, and which pair of measures actually carries the representation-vs-behaviour claim.
