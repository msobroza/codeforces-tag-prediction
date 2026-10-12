"""Settings of the label prior: what its recipe hashes (docs/adr/0004)."""

# Its registry key, and the file of its method folder that holds the tags it predicts.
NAME = "label_prior"
PRIOR_FILE = "label_prior.json"
# A focus tag is predicted, for every problem, when at least this many problems of the fit data
# have it. One is enough: predicting a tag for every problem scores F1 = 2p / (1 + p) on it, p
# being its share of positives, which beats the 0 of predicting nothing as soon as p is above 0.
MIN_POSITIVES = 1

# The values the run record and the recipe hold under "settings".
SETTINGS = {"min_positives": MIN_POSITIVES}
