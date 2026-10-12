"""The settings that shape an LLM's answer: the part of a request that its hash covers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

# How a prompted method's answer is returned: free text read by a fixed parser, or JSON that
# follows a strict schema. A prompt's versions are registered in this order.
OUTPUT_FORMATS: tuple[str, ...] = ("text", "json")


@dataclass(frozen=True)
class LlmSettings:
    """What shapes an answer, in a request to an LLM.

    Every field is part of a request's hash, so a changed setting never reuses an answer that
    was recorded under another. How a request is sent (concurrency, retries, timeout, whether
    the provider stores it) is not here: it stays in ``config.LLM_*``.

    Attributes:
        provider: A key of ``llm.providers.PROVIDERS``, e.g. ``"openai"``.
        model: The model's name at that provider.
        reasoning_effort: The reasoning effort to send, or None to send none.
        temperature: The sampling temperature, or None to send none.
        top_p: The nucleus sampling mass, or None to send none.
        max_output_tokens: The most tokens an answer may hold: a longer one is cut off.
        output_format: One of ``OUTPUT_FORMATS``.
    """

    provider: str
    model: str
    reasoning_effort: str | None
    temperature: float | None
    top_p: float | None
    max_output_tokens: int
    output_format: str

    def __post_init__(self) -> None:
        """Check the output format.

        Raises:
            ValueError: If ``output_format`` is not one of ``OUTPUT_FORMATS``.
        """
        if self.output_format not in OUTPUT_FORMATS:
            raise ValueError(
                f"{self.output_format!r} is not an output format: expected one of {OUTPUT_FORMATS}."
            )

    def to_dict(self) -> dict[str, object]:
        """Return the settings as a JSON-ready dict: one key per field, in field order."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> LlmSettings:
        """Rebuild settings from ``to_dict``'s output, e.g. as read back from JSON.

        Args:
            values: A value for every field, and no other key.

        Returns:
            The settings.

        Raises:
            TypeError: If a field is missing or a key is not a field.
            ValueError: If ``output_format`` is not one of ``OUTPUT_FORMATS``.
        """
        return cls(**values)
