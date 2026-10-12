"""The search's minibatches: learning-fold problems stratified on the focus tags (docs/adr/0010).

GEPA picks a minibatch before the piece it rewrites, so every minibatch holds positives of every
focus tag: first ``per_tag`` positives of each tag, drawn without replacement from that tag's
positives, then random other problems until ``size`` distinct problems. The sampler follows
GEPA's ``BatchSampler`` protocol structurally and imports nothing of gepa. GEPA's data ids are
positions into its trainset (``ListDataLoader``), so the sampler returns positions into
``sample_ids``, never sample ids. Its draw depends only on the seed and the iteration number, so
a replayed search draws the same minibatches.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import pandas as pd

from codeforces_tags.config import FOCUS_TAGS
from codeforces_tags.methods.typesafe_jev.settings import (
    SEARCH_MINIBATCH_SIZE,
    SEARCH_POSITIVES_PER_TAG,
    SEARCH_SEED,
)


class StratifiedSampler:
    """Draw each iteration's minibatch of learning-fold problems, stratified on the focus tags.

    Attributes:
        size: The distinct problems of a minibatch.
        per_tag: The positives of each focus tag a minibatch holds at least (all of a tag's
            positives when it has fewer).
        seed: The search seed; with the iteration number, it seeds each draw.
    """

    def __init__(
        self,
        sample_ids: Sequence[int],
        labels: pd.DataFrame,
        *,
        size: int = SEARCH_MINIBATCH_SIZE,
        per_tag: int = SEARCH_POSITIVES_PER_TAG,
        seed: int = SEARCH_SEED,
    ) -> None:
        """Index the positives of each focus tag among the problems GEPA learns from.

        Args:
            sample_ids: GEPA's trainset, in its order: a minibatch's positions index it.
            labels: 0/1 focus labels indexed by ``sample_id``, holding every id of
                ``sample_ids`` (other rows, e.g. of other folds, are never drawn).
            size: The distinct problems of a minibatch.
            per_tag: The positives of each focus tag drawn first.
            seed: The search seed.

        Raises:
            ValueError: If ``sample_ids`` repeats an id, ``labels`` lacks one of them,
                ``per_tag`` is negative, or ``size`` is smaller than the stratified draw
                (``per_tag`` times the focus tags) or larger than ``sample_ids``.
        """
        _check_sizes(len(sample_ids), size, per_tag)
        ids = [int(sample_id) for sample_id in sample_ids]
        if len(set(ids)) != len(ids):
            raise ValueError("sample_ids repeats a sample id")
        missing = sorted(set(ids) - {int(sample_id) for sample_id in labels.index})
        if missing:
            raise ValueError(f"labels has no row for sample ids {missing[:10]}")
        rows = labels.loc[ids, list(FOCUS_TAGS)].to_numpy()
        self._count = len(ids)
        self._positives = {
            tag: [position for position in range(len(ids)) if rows[position, column] == 1]
            for column, tag in enumerate(FOCUS_TAGS)
        }
        self.size = size
        self.per_tag = per_tag
        self.seed = seed

    def minibatch(self, iteration: int) -> list[int]:
        """Draw one iteration's minibatch.

        Args:
            iteration: GEPA's iteration number (``state.i``, 0 for the first iteration).

        Returns:
            ``size`` distinct positions into ``sample_ids``, ascending: ``per_tag`` positives
            of each focus tag (all of a tag's positives when it has fewer), then random others.
            The same seed and iteration always give the same positions.
        """
        rng = random.Random(f"{self.seed}:{iteration}")
        chosen: set[int] = set()
        for tag in FOCUS_TAGS:
            positives = self._positives[tag]
            chosen.update(rng.sample(positives, min(self.per_tag, len(positives))))
        others = [position for position in range(self._count) if position not in chosen]
        chosen.update(rng.sample(others, self.size - len(chosen)))
        return sorted(chosen)

    def next_minibatch_ids(self, loader: Any, state: Any) -> list[int]:
        """Give GEPA the minibatch of its current iteration (its ``BatchSampler`` protocol).

        GEPA's default sampling strategy asks once per iteration; a second call within one
        iteration gives the same minibatch.

        Args:
            loader: GEPA's trainset loader, whose ids are positions into ``sample_ids``.
            state: GEPA's state; its ``i`` is the current iteration.

        Returns:
            ``minibatch(state.i)``.

        Raises:
            ValueError: If the loader does not hold as many problems as ``sample_ids``.
        """
        if len(loader) != self._count:
            raise ValueError(f"the loader holds {len(loader)} problems, the sampler {self._count}")
        return self.minibatch(state.i)


def _check_sizes(count: int, size: int, per_tag: int) -> None:
    stratified = per_tag * len(FOCUS_TAGS)
    if per_tag < 0:
        raise ValueError(f"per_tag must not be negative, got {per_tag}")
    if size < stratified:
        raise ValueError(f"size {size} is smaller than the stratified draw of {stratified}")
    if size > count:
        raise ValueError(f"size {size} is larger than the {count} problems to draw from")
