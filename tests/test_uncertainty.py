"""Tests for prism.uncertainty — the Beta-belief math that everything rests on."""

from __future__ import annotations

import math
import random

import pytest

from prism.uncertainty import (
    Belief,
    _digamma,
    from_mean_std,
    probability_best,
    propagate,
    selection_entropy,
)


# --- summary statistics against closed forms -------------------------------

def test_belief_mean_variance_std_closed_form():
    b = Belief(1.0, 1.0)
    assert b.mean == pytest.approx(0.5)
    # Var of Beta(1,1) = ab/((a+b)^2 (a+b+1)) = 1/(4*3) = 1/12.
    assert b.variance == pytest.approx(1.0 / 12.0)
    assert b.std == pytest.approx(math.sqrt(1.0 / 12.0))


def test_belief_mean_general():
    b = Belief(4.0, 1.0)
    assert b.mean == pytest.approx(0.8)
    assert b.evidence == pytest.approx(5.0)


def test_belief_rejects_nonpositive_params():
    with pytest.raises(ValueError):
        Belief(0.0, 1.0)
    with pytest.raises(ValueError):
        Belief(1.0, -2.0)


# --- online update ----------------------------------------------------------

def test_updated_moves_posterior_and_is_immutable():
    b = Belief(1.0, 1.0)
    up = b.updated(1.0)  # full success -> alpha grows
    # Original untouched (immutability) and a new object is returned.
    assert b.alpha == 1.0 and b.beta == 1.0
    assert up is not b
    assert up.alpha == pytest.approx(2.0)
    assert up.beta == pytest.approx(1.0)
    assert up.mean > b.mean  # posterior moved toward success

    down = b.updated(0.0)  # full failure -> beta grows, mean drops
    assert down.mean < b.mean


def test_update_grows_evidence():
    b = Belief(1.0, 1.0)
    after = b.updated(0.7, weight=2.0)
    assert after.evidence == pytest.approx(b.evidence + 2.0)
    # 0.7 reward with weight 2 splits as +1.4 alpha / +0.6 beta.
    assert after.alpha == pytest.approx(1.0 + 1.4)
    assert after.beta == pytest.approx(1.0 + 0.6)


def test_update_clamps_reward():
    # Rewards outside [0,1] must be clamped, not blow up the params.
    hi = Belief(1.0, 1.0).updated(5.0)
    assert hi.alpha == pytest.approx(2.0) and hi.beta == pytest.approx(1.0)
    lo = Belief(1.0, 1.0).updated(-5.0)
    assert lo.alpha == pytest.approx(1.0) and lo.beta == pytest.approx(2.0)


# --- from_mean_std round-trip ----------------------------------------------

@pytest.mark.parametrize("mean,std", [(0.7, 0.10), (0.3, 0.08), (0.5, 0.15), (0.9, 0.05)])
def test_from_mean_std_round_trips(mean, std):
    b = from_mean_std(mean, std)
    assert b.mean == pytest.approx(mean, abs=1e-6)
    assert b.std == pytest.approx(std, abs=1e-6)


# --- propagation: uncertainty compounds ------------------------------------

def test_propagate_mean_is_product_of_means():
    up = Belief(8.0, 2.0)   # mean 0.8
    down = Belief(6.0, 4.0)  # mean 0.6
    comp = propagate(up, down)
    assert comp.mean == pytest.approx(up.mean * down.mean)


def test_propagate_relative_uncertainty_does_not_shrink():
    up = Belief(8.0, 2.0)
    down = Belief(8.0, 2.0)
    comp = propagate(up, down)
    cv_up = up.std / up.mean
    cv_down = down.std / down.mean
    cv_comp = comp.std / comp.mean
    # Coefficient of variation of a product is >= each input's — compounds.
    assert cv_comp >= cv_up - 1e-9
    assert cv_comp >= cv_down - 1e-9
    assert cv_comp > cv_up  # strictly larger here (both inputs uncertain)


# --- probability_best / selection_entropy ----------------------------------

def test_probability_best_sums_to_one_and_finds_dominant():
    rng = random.Random(0)
    beliefs = {"strong": Belief(90.0, 10.0), "weak": Belief(10.0, 90.0)}
    p = probability_best(beliefs, rng, samples=2000)
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["strong"] > 0.98
    assert p["weak"] < 0.02


def test_probability_best_empty():
    assert probability_best({}, random.Random(0)) == {}


def test_selection_entropy_certain_is_zero():
    assert selection_entropy({"a": 1.0, "b": 0.0}) == pytest.approx(0.0)


def test_selection_entropy_uniform_is_ln_k():
    k = 4
    uniform = {chr(ord("a") + i): 1.0 / k for i in range(k)}
    assert selection_entropy(uniform) == pytest.approx(math.log(k))


# --- entropy / digamma sanity ----------------------------------------------

def test_belief_entropy_finite_and_uniform_is_zero():
    # Differential entropy of Beta(1,1) == entropy of Uniform[0,1] == 0.
    assert Belief(1.0, 1.0).entropy == pytest.approx(0.0, abs=1e-9)
    # A concentrated belief has finite (negative) differential entropy.
    e = Belief(20.0, 5.0).entropy
    assert math.isfinite(e)
    assert e < 0.0


def test_digamma_known_values():
    euler_mascheroni = 0.5772156649015329
    assert _digamma(1.0) == pytest.approx(-euler_mascheroni, abs=1e-6)
    assert _digamma(2.0) == pytest.approx(1.0 - euler_mascheroni, abs=1e-6)
