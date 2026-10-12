"""The label prior: the F1 floor, and the worked example of adding a tagging method.

- ``settings``: its name, the file it saves, and the settings its recipe hashes.
- ``method``: ``LabelPrior``, the tagging method, and the module-level ``load`` the registry names.
- ``__main__``: ``python -m codeforces_tags.methods.label_prior freeze|verify``.

Importing this package imports none of them: a method's prediction path loads ``method`` only.
"""
