import json
from threading import Lock

from app.core.config import EVALUATION_DIR
from app.models.evaluation_models import EvaluationDecision


# -----------------------------------
# CONFIGURABLE WEIGHTS
# -----------------------------------

RETRIEVAL_WEIGHT = 0.20
SOURCE_RELEVANCE_WEIGHT = 0.20
GROUNDING_WEIGHT = 0.30
CORRECTNESS_WEIGHT = 0.20
CITATION_WEIGHT = 0.10


# -----------------------------------
# CONFIGURABLE THRESHOLDS
# -----------------------------------

AUTO_APPROVE_THRESHOLD = 0.85
REVIEW_RECOMMENDED_THRESHOLD = 0.65

CRITICAL_GROUNDING_THRESHOLD = 0.50
_CONFIG_LOCK = Lock()
_CONFIG_PATH = EVALUATION_DIR / "evaluation_config.json"


def evaluation_config() -> dict:
    return {
        "auto_approve_threshold": AUTO_APPROVE_THRESHOLD,
        "review_recommended_threshold": REVIEW_RECOMMENDED_THRESHOLD,
        "critical_grounding_threshold": CRITICAL_GROUNDING_THRESHOLD,
        "weights": {
            "retrieval_quality": RETRIEVAL_WEIGHT,
            "source_relevance": SOURCE_RELEVANCE_WEIGHT,
            "grounding": GROUNDING_WEIGHT,
            "answer_correctness": CORRECTNESS_WEIGHT,
            "citation_quality": CITATION_WEIGHT,
        },
    }


def configure_evaluation(values: dict) -> dict:
    """Validate and persist developer-controlled evaluation policy."""
    global AUTO_APPROVE_THRESHOLD, REVIEW_RECOMMENDED_THRESHOLD, CRITICAL_GROUNDING_THRESHOLD
    global RETRIEVAL_WEIGHT, SOURCE_RELEVANCE_WEIGHT, GROUNDING_WEIGHT, CORRECTNESS_WEIGHT, CITATION_WEIGHT
    thresholds = {
        key: float(values[key])
        for key in (
            "auto_approve_threshold",
            "review_recommended_threshold",
            "critical_grounding_threshold",
        )
        if key in values
    }
    weights = {key: float(value) for key, value in values.get("weights", {}).items()}
    if not all(0 <= value <= 1 for value in (*thresholds.values(), *weights.values())):
        raise ValueError("Evaluation thresholds and weights must be between 0 and 1.")
    if thresholds.get("auto_approve_threshold", AUTO_APPROVE_THRESHOLD) < thresholds.get("review_recommended_threshold", REVIEW_RECOMMENDED_THRESHOLD):
        raise ValueError("Auto-approve threshold must be at least the review threshold.")
    if weights and abs(sum(weights.get(key, current) for key, current in {
        "retrieval_quality": RETRIEVAL_WEIGHT,
        "source_relevance": SOURCE_RELEVANCE_WEIGHT,
        "grounding": GROUNDING_WEIGHT,
        "answer_correctness": CORRECTNESS_WEIGHT,
        "citation_quality": CITATION_WEIGHT,
    }.items()) - 1.0) > 0.001:
        raise ValueError("Evaluation weights must sum to 1.0.")
    with _CONFIG_LOCK:
        AUTO_APPROVE_THRESHOLD = thresholds.get("auto_approve_threshold", AUTO_APPROVE_THRESHOLD)
        REVIEW_RECOMMENDED_THRESHOLD = thresholds.get("review_recommended_threshold", REVIEW_RECOMMENDED_THRESHOLD)
        CRITICAL_GROUNDING_THRESHOLD = thresholds.get("critical_grounding_threshold", CRITICAL_GROUNDING_THRESHOLD)
        RETRIEVAL_WEIGHT = weights.get("retrieval_quality", RETRIEVAL_WEIGHT)
        SOURCE_RELEVANCE_WEIGHT = weights.get("source_relevance", SOURCE_RELEVANCE_WEIGHT)
        GROUNDING_WEIGHT = weights.get("grounding", GROUNDING_WEIGHT)
        CORRECTNESS_WEIGHT = weights.get("answer_correctness", CORRECTNESS_WEIGHT)
        CITATION_WEIGHT = weights.get("citation_quality", CITATION_WEIGHT)
        config = evaluation_config()
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return config


def _load_persisted_configuration() -> None:
    try:
        if _CONFIG_PATH.is_file():
            configure_evaluation(json.loads(_CONFIG_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError):
        # A corrupt developer configuration must not prevent the API from
        # starting; safe defaults remain active.
        return


_load_persisted_configuration()


def calculate_overall_confidence(
    retrieval_quality: float,
    source_relevance: float,
    grounding: float,
    answer_correctness: float,
    citation_quality: float,
) -> float:
    """
    Calculates weighted confidence from
    independent evaluation dimensions.
    """

    confidence = (
        retrieval_quality * RETRIEVAL_WEIGHT
        + source_relevance * SOURCE_RELEVANCE_WEIGHT
        + grounding * GROUNDING_WEIGHT
        + answer_correctness * CORRECTNESS_WEIGHT
        + citation_quality * CITATION_WEIGHT
    )

    return round(
        max(0.0, min(1.0, confidence)),
        4
    )


def make_decision(
    overall_confidence: float,
    grounding: float,
) -> EvaluationDecision:
    """
    Converts evaluation scores into a
    production decision.

    Grounding acts as a hard safety gate.
    """

    # -----------------------------------
    # HARD GROUNDING GATE
    # -----------------------------------

    if grounding < CRITICAL_GROUNDING_THRESHOLD:
        return EvaluationDecision.HUMAN_REVIEW

    # -----------------------------------
    # OVERALL CONFIDENCE
    # -----------------------------------

    if overall_confidence >= AUTO_APPROVE_THRESHOLD:
        return EvaluationDecision.AUTO_APPROVE

    if overall_confidence >= REVIEW_RECOMMENDED_THRESHOLD:
        return EvaluationDecision.REVIEW_RECOMMENDED

    return EvaluationDecision.HUMAN_REVIEW
