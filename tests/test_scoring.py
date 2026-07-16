"""Tests for prism.scoring — calibrated (empirical-Bayes) winner selection."""

from __future__ import annotations

import pytest

from prism.scoring import CalibratedScore, calibrate, select_winner
from prism.uncertainty import Belief


def test_high_raw_with_strong_prior_yields_high_posterior():
    prior = Belief(20.0, 2.0)  # mean ~0.909, already confident it's great
    cs = calibrate("a", raw_score=0.9, prior=prior, scorer_reliability=1.0)
    assert cs.posterior_mean > 0.8
    assert cs.prior_mean == pytest.approx(prior.mean)
    assert cs.raw_score == pytest.approx(0.9)


def test_low_reliability_shrinks_toward_prior_more_than_high_reliability():
    prior = Belief(2.0, 8.0)  # mean 0.2 — we think this agent is weak
    raw = 0.9  # a surprisingly high (possibly lucky) reading

    high = calibrate("a", raw, prior, scorer_reliability=1.0)
    low = calibrate("a", raw, prior, scorer_reliability=0.5)

    # A trusted scorer moves the posterior further from the prior; a doubtful
    # scorer's evidence is discounted, leaving the posterior closer to the prior.
    assert high.posterior_mean > low.posterior_mean
    dist_high = abs(high.posterior_mean - prior.mean)
    dist_low = abs(low.posterior_mean - prior.mean)
    assert dist_low < dist_high


def test_useless_scorer_leaves_posterior_at_prior():
    prior = Belief(2.0, 8.0)
    cs = calibrate("a", raw_score=0.99, prior=prior, scorer_reliability=0.25)
    # reliability 0.25 -> weight 0 -> no movement.
    assert cs.posterior_mean == pytest.approx(prior.mean)


def test_select_winner_picks_max_posterior():
    scored = [
        CalibratedScore("a", raw_score=0.5, posterior_mean=0.60, prior_mean=0.5),
        CalibratedScore("b", raw_score=0.9, posterior_mean=0.85, prior_mean=0.5),
        CalibratedScore("c", raw_score=0.7, posterior_mean=0.72, prior_mean=0.5),
    ]
    assert select_winner(scored).agent_name == "b"


def test_select_winner_tiebreak_is_deterministic():
    # Equal posterior and raw score -> tie-break on name (smaller ord wins).
    a = CalibratedScore("aa", raw_score=0.8, posterior_mean=0.7, prior_mean=0.5)
    z = CalibratedScore("zz", raw_score=0.8, posterior_mean=0.7, prior_mean=0.5)
    w1 = select_winner([a, z])
    w2 = select_winner([z, a])  # order must not matter
    assert w1.agent_name == "aa"
    assert w2.agent_name == "aa"


def test_select_winner_raw_score_breaks_posterior_tie():
    a = CalibratedScore("a", raw_score=0.6, posterior_mean=0.7, prior_mean=0.5)
    b = CalibratedScore("b", raw_score=0.9, posterior_mean=0.7, prior_mean=0.5)
    assert select_winner([a, b]).agent_name == "b"
