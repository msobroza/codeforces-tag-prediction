"""The LLM toolkit: what every prompted method shares (docs/adr/0003).

A prompted method is a tagging method that asks an LLM through a fixed prompt. The toolkit is
provider-agnostic: a provider's SDK is imported only inside its adapter's ``ask``, and only
``__main__`` imports mlflow. No test sends a request: each injects a fake ``Ask``.

- ``settings``: ``LlmSettings``, the settings that shape an answer, which a request's hash covers.
- ``prompts``: ``PromptSnapshot``, ``LlmMethodSpec`` and ``build_requests``, which renders a
  request, and its hash, for each problem.
- ``client``: ``LlmAnswer`` and ``Ask``, the seam between the toolkit and a provider: what one
  request returns, and the function that sends it.
- ``providers``: one adapter module per provider (OpenAI's first), listed in ``PROVIDERS``.
- ``answers``: the fixed parsers of an answer's text, the 0/1 encoding, the summary, the cost
  and the rows of ``per_problem.csv``.
- ``store``: the answers files, append-only JSON lines that are both the resume cache and the
  record.
- ``runner``: ``answer_each``, which sends requests concurrently under a cost cap.
- ``prompted``: ``PromptedLlm``, the tagging method; ``out_of_fold``: ``cross_requests``, the
  dev requests of a prompted method that learns from dev labels.
- ``__main__``: ``python -m codeforces_tags.llm render|run|score|verify``.

Importing this package sets the variables MLflow reads when it is imported
(``MLFLOW_IMPORT_ENV``: no telemetry, no hint to coding agents), before ``__main__`` imports it.
"""

from codeforces_tags.config import MLFLOW_IMPORT_ENV
from codeforces_tags.tracking.environment import mlflow_env

mlflow_env(MLFLOW_IMPORT_ENV)
