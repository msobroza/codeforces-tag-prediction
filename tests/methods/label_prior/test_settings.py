"""Tests for the label prior's settings."""

from codeforces_tags.methods.label_prior.settings import MIN_POSITIVES, SETTINGS


def test_settings_hold_the_values_the_recipe_hashes():
    # One positive is enough: predicting a tag for every problem scores F1 = 2p / (1 + p), which
    # is above the 0 of predicting nothing as soon as p, the share of positives, is above 0.
    assert MIN_POSITIVES == 1
    assert SETTINGS == {"min_positives": MIN_POSITIVES}
