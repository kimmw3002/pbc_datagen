"""Tests for pt_update_labels, pt_accumulate_histograms, pt_count_round_trips.

Steps 1.5.3–1.5.5: pure index arithmetic on int arrays, no models needed.

Label convention:
  LABEL_NONE = 0   — no label yet
  LABEL_UP   = 1   — last visited T_min (cold end, slot 0)
  LABEL_DOWN = -1  — last visited T_max (hot end, slot M-1)
"""

from __future__ import annotations

LABEL_NONE = 0
LABEL_UP = 1
LABEL_DOWN = -1


# ── 1.5.3  pt_update_labels ──────────────────────────────────────────


class TestUpdateLabels:
    def test_sets_up_at_cold_down_at_hot(self):
        """Replica at slot 0 gets UP, replica at slot M-1 gets DOWN."""
        from pbc_datagen._core import pt_update_labels

        M = 4
        # t2r: replica 2 sits at slot 0, replica 1 sits at slot 3
        t2r = [2, 0, 3, 1]
        labels = [LABEL_NONE] * M

        pt_update_labels(labels, t2r, M)

        assert labels[2] == LABEL_UP  # replica at coldest slot
        assert labels[1] == LABEL_DOWN  # replica at hottest slot
        assert labels[0] == LABEL_NONE  # untouched
        assert labels[3] == LABEL_NONE  # untouched

    def test_overwrites_previous_label(self):
        """A replica that was DOWN and reaches slot 0 becomes UP."""
        from pbc_datagen._core import pt_update_labels

        M = 3
        t2r = [1, 2, 0]
        labels = [LABEL_NONE, LABEL_DOWN, LABEL_UP]  # replica 1 was DOWN

        pt_update_labels(labels, t2r, M)

        assert labels[1] == LABEL_UP  # overwritten: now at cold end
        assert labels[0] == LABEL_DOWN  # overwritten: now at hot end
        assert labels[2] == LABEL_UP  # unchanged (was UP, not at extreme)


# ── 1.5.4  pt_accumulate_histograms ──────────────────────────────────


class TestAccumulateHistograms:
    def test_increments_matching_counters(self):
        """n_up incremented for UP-labeled replicas, n_down for DOWN."""
        from pbc_datagen._core import pt_accumulate_histograms

        M = 4
        # t2r[0]=0, t2r[1]=1, t2r[2]=2, t2r[3]=3 (identity)
        t2r = [0, 1, 2, 3]
        labels = [LABEL_UP, LABEL_DOWN, LABEL_UP, LABEL_DOWN]
        n_up = [0, 0, 0, 0]
        n_down = [0, 0, 0, 0]

        pt_accumulate_histograms(n_up, n_down, labels, t2r, M)

        assert n_up == [1, 0, 1, 0]
        assert n_down == [0, 1, 0, 1]

    def test_skips_label_none(self):
        """Slots with LABEL_NONE replicas don't increment either counter."""
        from pbc_datagen._core import pt_accumulate_histograms

        M = 3
        t2r = [0, 1, 2]
        labels = [LABEL_NONE, LABEL_UP, LABEL_NONE]
        n_up = [0, 0, 0]
        n_down = [0, 0, 0]

        pt_accumulate_histograms(n_up, n_down, labels, t2r, M)

        assert n_up == [0, 1, 0]
        assert n_down == [0, 0, 0]

    def test_accumulates_over_calls(self):
        """Counters accumulate across multiple calls."""
        from pbc_datagen._core import pt_accumulate_histograms

        M = 2
        t2r = [0, 1]
        labels = [LABEL_UP, LABEL_DOWN]
        n_up = [0, 0]
        n_down = [0, 0]

        for _ in range(5):
            pt_accumulate_histograms(n_up, n_down, labels, t2r, M)

        assert n_up == [5, 0]
        assert n_down == [0, 5]


# ── 1.5.5  pt_count_round_trips ──────────────────────────────────────


class TestCountRoundTrips:
    def test_detects_up_to_down_at_hot_end(self):
        """UP-labeled replica reaching slot M-1 and becoming DOWN = 1 round trip."""
        from pbc_datagen._core import pt_count_round_trips

        M = 4
        t2r = [0, 1, 2, 3]

        prev_labels = [LABEL_NONE, LABEL_NONE, LABEL_NONE, LABEL_UP]
        curr_labels = [LABEL_NONE, LABEL_NONE, LABEL_NONE, LABEL_DOWN]

        assert pt_count_round_trips(curr_labels, prev_labels, t2r, M) == 1

    def test_returns_zero_when_no_transition(self):
        """No UP→DOWN transition at hot end → 0."""
        from pbc_datagen._core import pt_count_round_trips

        M = 4
        t2r = [0, 1, 2, 3]

        # DOWN stays DOWN — not a new trip
        prev_labels = [LABEL_NONE, LABEL_NONE, LABEL_NONE, LABEL_DOWN]
        curr_labels = [LABEL_NONE, LABEL_NONE, LABEL_NONE, LABEL_DOWN]
        assert pt_count_round_trips(curr_labels, prev_labels, t2r, M) == 0

        # NONE → DOWN — not a trip (never visited cold end)
        prev_labels = [LABEL_NONE, LABEL_NONE, LABEL_NONE, LABEL_NONE]
        curr_labels = [LABEL_NONE, LABEL_NONE, LABEL_NONE, LABEL_DOWN]
        assert pt_count_round_trips(curr_labels, prev_labels, t2r, M) == 0

    def test_returns_zero_when_up_not_at_hot_end(self):
        """UP→DOWN transition at a non-extreme slot doesn't count."""
        from pbc_datagen._core import pt_count_round_trips

        M = 4
        t2r = [0, 1, 2, 3]

        # Replica 2 transitions UP→DOWN, but it's not at slot M-1
        prev_labels = [LABEL_NONE, LABEL_NONE, LABEL_UP, LABEL_NONE]
        curr_labels = [LABEL_NONE, LABEL_NONE, LABEL_DOWN, LABEL_DOWN]
        assert pt_count_round_trips(curr_labels, prev_labels, t2r, M) == 0
