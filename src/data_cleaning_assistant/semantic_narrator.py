from __future__ import annotations

import json
from typing import Protocol

from pydantic import BaseModel, Field

from .ai_provider import StructuredAIProvider
from .presentation import ActionContextPayload


class SemanticNarration(BaseModel):
    title: str = Field(max_length=140)
    explanation: str = Field(max_length=700)
    data_context: str = Field(max_length=700)
    caution: str = Field(max_length=500)


class ActionNarrator(Protocol):
    model: str

    def narrate(self, payload: ActionContextPayload) -> SemanticNarration:
        ...


class LocalSemanticNarrator:
    """Advisory local-AI explanation layer with no transformation authority."""

    def __init__(
        self,
        provider: StructuredAIProvider,
        *,
        model: str,
    ) -> None:
        self.provider = provider
        self.model = model

    def narrate(self, payload: ActionContextPayload) -> SemanticNarration:
        system_prompt = (
            "You explain a deterministic data-cleaning proposal to a researcher. "
            "You have ZERO authority to propose, modify, approve, reject, broaden, or "
            "execute a transformation. The cleaning rule has already been created by "
            "deterministic code. Your only job is to rewrite its explanation in clear, "
            "dataset-aware language. Infer cautiously from column names, neighboring "
            "columns, inferred type, and example values. Never claim a domain meaning "
            "as fact unless the supplied evidence makes it explicit. Use phrases such "
            "as 'appears to', 'likely', or 'looks like' when interpreting semantics. "
            "Do not introduce new mappings, missing tokens, formats, formulas, or edits. "
            "Do not mention JSON, code, action types, schemas, or implementation details. "
            "Write as a careful analyst who is familiar with the visible dataset."
        )
        user_prompt = json.dumps(
            {
                "dataset": payload.dataset_name,
                "table": payload.table,
                "column": payload.column,
                "nearby_columns": payload.nearby_columns,
                "inferred_type": payload.inferred_type,
                "sample_values": payload.sample_values,
                "deterministic_title": payload.deterministic_title,
                "deterministic_summary": payload.deterministic_summary,
                "before_after_examples": payload.before_after_examples,
                "instructions": {
                    "title": (
                        "One concise plain-English title that describes the actual "
                        "review decision in the dataset's own terms."
                    ),
                    "explanation": (
                        "Explain what will change and how many conceptual kinds of values "
                        "are involved. Do not add transformations."
                    ),
                    "data_context": (
                        "Explain what this column appears to represent based only on the "
                        "supplied evidence and why consistency may matter to someone using it."
                    ),
                    "caution": (
                        "State the most important human check before approval. Do not invent "
                        "a new risk unrelated to the supplied proposal."
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        return self.provider.generate_structured(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=SemanticNarration,
        )
