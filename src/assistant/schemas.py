from __future__ import annotations

from typing import Any,Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
    

class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(..., min_length=1)
    product_ids: list[str] = Field(..., min_length=1)

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_id must be a non-empty string")
        return v

    @field_validator("product_ids")
    @classmethod
    def validate_product_ids(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in values:
            if not isinstance(item, str):
                raise ValueError("each product_id must be a string")
            item = item.strip()
            if not item:
                raise ValueError("product_ids must not contain empty strings")
            cleaned.append(item)

        if not cleaned:
            raise ValueError("product_ids must contain at least one item")

        seen: set[str] = set()
        deduped: list[str] = []
        for pid in cleaned:
            if pid not in seen:
                seen.add(pid)
                deduped.append(pid)

        return deduped


class ShoppingQuery(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description="Product category such as Fashion, Electronics, FMCG, headphones, shoes",
    )
    max_price: Optional[float] = Field(
        default=None,
        description="Maximum user budget in Rs.",
    )
    brand: Optional[str] = Field(
        default=None,
        description="Preferred brand if mentioned",
    )
    intent: str = Field(default="recommend", description="User intent")
    notes: Optional[str] = Field(
        default=None,
        description="Any additional user preferences",
    )

    @field_validator("category", "brand", "notes")
    @classmethod
    def strip_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("max_price")
    @classmethod
    def validate_max_price(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value <= 0:
            raise ValueError("max_price must be positive")
        return value


class RecommendationResult(BaseModel):
    product_id: str
    title: str
    category: str
    brand: str
    price: float
    score: float
    reasons: list[str]


class AssistantResponse(BaseModel):
    user_id: str
    interpreted_query: ShoppingQuery
    recommendations: list[RecommendationResult]
    assistant_message: str


class ConversationSession(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str
    memory: list[Any] = Field(default_factory=list, exclude=True)