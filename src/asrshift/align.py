"""Word alignment with support for wildcard reference tokens.

A plain Levenshtein alignment (substitution, deletion and insertion all cost 1) extended with one
rule: a reference token equal to ``WILDCARD`` matches any run of zero or more hypothesis tokens at
zero cost. Without wildcards the edit count equals the standard word-level edit distance used by
jiwer, which the tests check.

Each reference token gets a status ``C`` (correct), ``S`` (substituted), ``D`` (deleted) or ``W``
(wildcard). Each hypothesis token gets ``C``, ``S``, ``I`` (inserted) or ``W`` (absorbed by a
wildcard). Word-level labels and entity scoring are built from these statuses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from asrshift.textnorm import WILDCARD


@dataclass
class Alignment:
    ref: list[str]
    hyp: list[str]
    ref_status: list[str] = field(default_factory=list)
    hyp_status: list[str] = field(default_factory=list)
    # hyp index aligned to each ref token (-1 if none)
    ref_to_hyp: list[int] = field(default_factory=list)
    # number of hypothesis tokens absorbed by each wildcard, left to right
    wild_runs: list[int] = field(default_factory=list)

    def wild_excess(self, cap: int = 3) -> int:
        """Absorbed tokens beyond ``cap`` per wildcard. Counting them as insertions gives an upper
        bound on the score a capped wildcard would give (exact when the excess is zero)."""
        return sum(max(0, r - cap) for r in self.wild_runs)

    @property
    def n_ref(self) -> int:
        """Number of scorable reference words (wildcards excluded)."""
        return sum(1 for s in self.ref_status if s != "W")

    @property
    def hits(self) -> int:
        return self.ref_status.count("C")

    @property
    def substitutions(self) -> int:
        return self.ref_status.count("S")

    @property
    def deletions(self) -> int:
        return self.ref_status.count("D")

    @property
    def insertions(self) -> int:
        return self.hyp_status.count("I")

    @property
    def absorbed(self) -> int:
        return self.hyp_status.count("W")

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        n = self.n_ref
        if n == 0:
            return float("nan")
        return self.errors / n


def align(ref: list[str], hyp: list[str]) -> Alignment:
    n, m = len(ref), len(hyp)
    wild = np.array([t == WILDCARD for t in ref], dtype=bool)
    hyp_arr = np.array(hyp, dtype=object)
    ar = np.arange(m + 1)

    D = np.empty((n + 1, m + 1), dtype=np.int64)
    D[0] = ar
    for i in range(1, n + 1):
        prev = D[i - 1]
        if wild[i - 1]:
            D[i] = np.minimum.accumulate(prev)
            continue
        row = np.empty(m + 1, dtype=np.int64)
        row[0] = prev[0] + 1
        if m:
            cost = (hyp_arr != ref[i - 1]).astype(np.int64)
            tmp = np.minimum(prev[:-1] + cost, prev[1:] + 1)
            # row[j] = min(tmp[j-1], row[j-1] + 1)  ->  prefix-min trick over (tmp[k] - k) + j
            cand = np.minimum.accumulate(np.concatenate(([row[0]], tmp)) - ar) + ar
            row[1:] = cand[1:]
        D[i] = row

    ref_status = [""] * n
    ref_to_hyp = [-1] * n
    hyp_status = [""] * m
    wild_runs: list[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and wild[i - 1]:
            ref_status[i - 1] = "W"
            # absorb as few hypothesis tokens as possible: largest k with D[i-1][k] == D[i][j]
            k = j
            while D[i - 1][k] != D[i][j]:
                k -= 1
            for t in range(k, j):
                hyp_status[t] = "W"
            wild_runs.append(j - k)
            i, j = i - 1, k
            continue
        if i > 0 and j > 0:
            same = ref[i - 1] == hyp[j - 1]
            if D[i][j] == D[i - 1][j - 1] + (0 if same else 1):
                ref_status[i - 1] = "C" if same else "S"
                hyp_status[j - 1] = "C" if same else "S"
                ref_to_hyp[i - 1] = j - 1
                i, j = i - 1, j - 1
                continue
        if i > 0 and D[i][j] == D[i - 1][j] + 1:
            ref_status[i - 1] = "D"
            i -= 1
            continue
        hyp_status[j - 1] = "I"
        j -= 1

    return Alignment(ref=ref, hyp=hyp, ref_status=ref_status, hyp_status=hyp_status, ref_to_hyp=ref_to_hyp,
                     wild_runs=wild_runs[::-1])
