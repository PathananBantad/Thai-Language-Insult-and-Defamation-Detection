from typing import List, Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=200)
    next_path: Optional[str] = Field(default=None, max_length=200)


class AuthStatusResponse(BaseModel):
    authenticated: bool
    username: Optional[str] = None
    protect_analyzer: bool = False


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Comment text to classify")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class BinaryInsultDefamationRequest(BaseModel):
    # Empty values are accepted here so the binary service can return an
    # explicit abstention result rather than silently coercing the input.
    text: str = Field(default="", max_length=5000, description="Raw text for Insult/Defamation screening")


class TargetEntity(BaseModel):
    text: str
    type: str
    start: int
    end: int


class TargetInfo(BaseModel):
    status: str
    entities: List[TargetEntity] = Field(default_factory=list)


class AllegationCue(BaseModel):
    text: str = ""
    category: str = ""


class AllegationAnalysisResult(BaseModel):
    result: str
    hedged: bool = False
    cues: List[AllegationCue] = Field(default_factory=list)
    rule_based_only: bool = True


class EvidenceSpan(BaseModel):
    kind: str
    text: str
    start: int
    end: int
    source_text: str = "raw"


class BinaryInsultDefamationResponse(BaseModel):
    status: str
    reason: Optional[str] = None
    predicted_label: Optional[str] = None
    probabilities: Optional[dict[str, float]] = None
    confidence: Optional[float] = None
    target: TargetInfo
    allegation_analysis: AllegationAnalysisResult
    evidence_spans: List[EvidenceSpan] = Field(default_factory=list)
    highlighting_note: Optional[str] = None
    explanation: str
    disclaimer: str


class PredictionResponse(BaseModel):
    request_id: str
    text: str
    processed_text: str
    predicted_label: str
    toxic_score: float
    confidence: float
    threshold: float
    recommendation: str
    source_model: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


class ReviewedExampleRequest(BaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    text: str = Field(..., min_length=1, max_length=5000)
    reviewed_label: str = Field(..., pattern="^(neg|neu)$")
    predicted_label: Optional[str] = Field(default=None, max_length=20)
    toxicity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source_model: Optional[str] = Field(default=None, max_length=200)


class ReviewedExampleResponse(BaseModel):
    status: str
    reviewed_count: int
    path: str
