
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import query as query_route
from app.models.evaluation_models import (
    EvaluationDecision,
    EvaluationResult,
)


client = TestClient(app)


# ---------------------------------------------------------
# TEST AUTHENTICATION HELPERS
# ---------------------------------------------------------

def get_test_token():
    """Create/reuse a dedicated test user without hitting auth rate limits."""
    from app.core.database import SessionLocal
    from app.models.auth_models import User
    from app.services.security.auth_service import (
        create_access_token,
        create_user,
    )

    email = "query_evaluation_test@example.com"
    password = "TestPass123!"

    db = SessionLocal()

    try:
        user = db.query(User).filter(User.email == email).first()

        if user is None:
            user = create_user(
                db,
                email=email,
                password=password,
            )

        return create_access_token(user.id)
    finally:
        db.close()


def auth_headers():
    return {
        "Authorization": f"Bearer {get_test_token()}",
    }


def get_test_user_token(email: str):
    """Create/reuse a dedicated test user and return a JWT."""
    from app.core.database import SessionLocal
    from app.models.auth_models import User
    from app.services.security.auth_service import (
        create_access_token,
        create_user,
    )

    password = "TestPass123!"
    db = SessionLocal()

    try:
        user = db.query(User).filter(User.email == email).first()

        if user is None:
            user = create_user(
                db,
                email=email,
                password=password,
            )

        return create_access_token(user.id)
    finally:
        db.close()


def auth_headers_for(email: str):
    return {
        "Authorization": f"Bearer {get_test_user_token(email)}",
    }


def allow_semantic_security(monkeypatch):
    from app.models.security_models import (
        SecurityCheckResult,
        SecurityDecision,
    )

    monkeypatch.setattr(
        query_route,
        "classify_question_semantically",
        lambda question, api_key: SecurityCheckResult(
            decision=SecurityDecision.ALLOW,
            is_safe=True,
            risk_score=0.0,
            reasons=["Test request explicitly allowed."],
        ),
    )


def get_test_user_id(email: str):
    """Return the database ID for a dedicated test user."""
    from app.core.database import SessionLocal
    from app.models.auth_models import User

    db = SessionLocal()

    try:
        user = db.query(User).filter(User.email == email).first()

        if user is None:
            raise AssertionError(
                f"Test user was not created: {email}"
            )

        return user.id
    finally:
        db.close()


# ---------------------------------------------------------
# MOCK RETRIEVAL OUTPUT
# ---------------------------------------------------------

def build_mock_retrieval_output():
    return {
        "status": "success",
        "active_document": "test_document.pdf",
        "confidence": "high",
        "best_score": 0.8,
        "average_score": 1.0,
        "retrieved_chunks": [
            {
                "chunk_id": 1,
                "file_name": "test_document.pdf",
                "page_number": 3,
                "text": (
                    "The project uses hybrid retrieval "
                    "combining vector search and BM25."
                ),
                "score": 0.8,
            },
            {
                "chunk_id": 2,
                "file_name": "test_document.pdf",
                "page_number": 4,
                "text": (
                    "The system reranks retrieved evidence "
                    "before generating the final answer."
                ),
                "score": 1.0,
            },
        ],
    }


# ---------------------------------------------------------
# MOCK EVALUATION
# ---------------------------------------------------------

def build_mock_evaluation():
    return EvaluationResult(
        evaluation_id="eval-test-001",
        user_id=None,
        question="How does the system retrieve evidence?",
        answer="The system uses hybrid retrieval.",
        retrieval_quality=0.90,
        source_relevance=0.90,
        grounding=1.00,
        answer_correctness=0.90,
        citation_quality=1.00,
        overall_confidence=0.94,
        decision=EvaluationDecision.AUTO_APPROVE,
        reasons=[
            "All evaluation dimensions are within acceptable ranges."
        ],
        claims=[],
    )


# ---------------------------------------------------------
# EVALUATION SAVE MOCK
# ---------------------------------------------------------

def mock_evaluation_storage(monkeypatch):
    """Capture saved evaluations and verify user ownership."""
    saved_evaluations = []

    def mock_save_evaluation(evaluation, user_id=None):
        saved_evaluations.append(
            {
                "evaluation": evaluation,
                "user_id": user_id,
            }
        )
        return evaluation.evaluation_id

    monkeypatch.setattr(
        query_route,
        "save_evaluation",
        mock_save_evaluation,
    )

    return saved_evaluations


def assert_evaluation_saved(
    saved_evaluations,
    evaluation,
    expected_email="query_evaluation_test@example.com",
):
    assert len(saved_evaluations) == 1

    saved = saved_evaluations[0]

    assert saved["evaluation"] is evaluation
    assert saved["user_id"] == get_test_user_id(expected_email)


# ---------------------------------------------------------
# TEST 1: AUTO APPROVE
# ---------------------------------------------------------

def test_ask_route_integrates_evaluation(monkeypatch):
    allow_semantic_security(monkeypatch)

    monkeypatch.setattr(
        query_route,
        "resolve_authorized_api_key",
        lambda user, access_password, user_openai_api_key: "test-api-key",
    )

    monkeypatch.setattr(
        query_route,
        "retrieve_relevant_chunks",
        lambda question, user_id=None: build_mock_retrieval_output(),
    )

    monkeypatch.setattr(
        query_route,
        "get_active_document",
        lambda user_id=None: {"active_doc_id": "test-doc-001"},
    )

    monkeypatch.setattr(
        query_route,
        "generate_grounded_answer",
        lambda question, retrieved_chunks, api_key: (
            "The system uses hybrid retrieval."
        ),
    )

    mock_evaluation = build_mock_evaluation()

    monkeypatch.setattr(
        query_route,
        "evaluate_rag_response",
        lambda question, answer, retrieved_chunks, evidence, api_key, use_llm_judges: (
            mock_evaluation
        ),
    )

    saved_evaluations = mock_evaluation_storage(monkeypatch)

    response = client.post(
        "/api/ask",
        headers=auth_headers(),
        json={
            "question": "How does the system retrieve evidence?",
            "access_password": "test-password",
            "user_openai_api_key": "",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert data["question"] == "How does the system retrieve evidence?"
    assert data["answer"] == "The system uses hybrid retrieval."

    assert "evaluation" in data

    assert data["evaluation"]["evaluation_id"] == "eval-test-001"
    assert data["evaluation"]["retrieval_quality"] == 0.90
    assert data["evaluation"]["source_relevance"] == 0.90
    assert data["evaluation"]["grounding"] == 1.0
    assert data["evaluation"]["answer_correctness"] == 0.90
    assert data["evaluation"]["citation_quality"] == 1.0
    assert data["evaluation"]["overall_confidence"] == 0.94
    assert data["evaluation"]["decision"] == "AUTO_APPROVE"

    assert_evaluation_saved(
        saved_evaluations,
        mock_evaluation,
    )


# ---------------------------------------------------------
# TEST 2: LOW RETRIEVAL CONFIDENCE
# ---------------------------------------------------------

def test_ask_route_does_not_call_evaluation_when_retrieval_is_low(
    monkeypatch,
):
    allow_semantic_security(monkeypatch)

    monkeypatch.setattr(
        query_route,
        "resolve_authorized_api_key",
        lambda user, access_password, user_openai_api_key: "test-api-key",
    )

    monkeypatch.setattr(
        query_route,
        "retrieve_relevant_chunks",
        lambda question, user_id=None: {
            "status": "success",
            "active_document": "test_document.pdf",
            "confidence": "low",
            "best_score": 3.0,
            "average_score": 3.2,
            "retrieved_chunks": [],
        },
    )

    evaluation_called = False

    def fail_if_evaluation_called(*args, **kwargs):
        nonlocal evaluation_called
        evaluation_called = True

        raise AssertionError(
            "Evaluation should not run when retrieval confidence is low."
        )

    monkeypatch.setattr(
        query_route,
        "evaluate_rag_response",
        fail_if_evaluation_called,
    )

    response = client.post(
        "/api/ask",
        headers=auth_headers(),
        json={
            "question": "Question outside the uploaded PDF",
            "access_password": "test-password",
            "user_openai_api_key": "",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "error"
    assert data["answer"] is None
    assert data["evaluation"] is None
    assert evaluation_called is False


# ---------------------------------------------------------
# TEST 3: GET EVALUATION
# ---------------------------------------------------------

def test_get_evaluation_returns_persisted_evaluation(monkeypatch):
    mock_evaluation = build_mock_evaluation()

    monkeypatch.setattr(
        query_route,
        "load_evaluation",
        lambda evaluation_id, user_id=None: (
            mock_evaluation
            if evaluation_id == "eval-test-001"
            else None
        ),
    )

    response = client.get(
        "/api/evaluation/eval-test-001",
        headers=auth_headers(),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["evaluation_id"] == "eval-test-001"
    assert data["question"] == "How does the system retrieve evidence?"
    assert data["answer"] == "The system uses hybrid retrieval."
    assert data["retrieval_quality"] == 0.90
    assert data["source_relevance"] == 0.90
    assert data["grounding"] == 1.0
    assert data["answer_correctness"] == 0.90
    assert data["citation_quality"] == 1.0
    assert data["overall_confidence"] == 0.94
    assert data["decision"] == "AUTO_APPROVE"


def test_get_evaluation_returns_404_when_not_found(monkeypatch):
    monkeypatch.setattr(
        query_route,
        "load_evaluation",
        lambda evaluation_id, user_id=None: None,
    )

    response = client.get(
        "/api/evaluation/eval-does-not-exist",
        headers=auth_headers(),
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Evaluation not found."


# ---------------------------------------------------------
# TEST 4: HUMAN REVIEW SUBMISSION
# ---------------------------------------------------------

def test_submit_human_review_saves_review(monkeypatch):
    mock_evaluation = build_mock_evaluation()

    monkeypatch.setattr(
        query_route,
        "load_evaluation",
        lambda evaluation_id, user_id=None: (
            mock_evaluation
            if evaluation_id == "eval-test-001"
            else None
        ),
    )

    saved_reviews = []

    def mock_save_human_review(review, user_id=None):
        saved_reviews.append(review)
        return review.evaluation_id

    monkeypatch.setattr(
        query_route,
        "save_human_review",
        mock_save_human_review,
    )

    response = client.post(
        "/api/evaluation/eval-test-001/review",
        headers=auth_headers(),
        json={
            "decision": "HUMAN_REVIEW",
            "feedback_category": "incorrect_answer",
            "feedback_text": "The answer needs correction.",
            "corrected_answer": "The corrected answer.",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert data["evaluation_id"] == "eval-test-001"

    assert data["review"]["evaluation_id"] == "eval-test-001"
    assert data["review"]["decision"] == "HUMAN_REVIEW"
    assert data["review"]["feedback_category"] == "incorrect_answer"
    assert data["review"]["feedback_text"] == (
        "The answer needs correction."
    )
    assert data["review"]["corrected_answer"] == (
        "The corrected answer."
    )

    assert len(saved_reviews) == 1

    saved_review = saved_reviews[0]

    assert saved_review.evaluation_id == "eval-test-001"
    assert saved_review.decision.value == "HUMAN_REVIEW"
    assert saved_review.feedback_category.value == "incorrect_answer"
    assert saved_review.feedback_text == (
        "The answer needs correction."
    )
    assert saved_review.corrected_answer == (
        "The corrected answer."
    )


def test_submit_human_review_returns_404_when_evaluation_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        query_route,
        "load_evaluation",
        lambda evaluation_id, user_id=None: None,
    )

    response = client.post(
        "/api/evaluation/eval-does-not-exist/review",
        headers=auth_headers(),
        json={
            "decision": "HUMAN_REVIEW",
            "feedback_category": "incorrect_answer",
            "feedback_text": "The evaluation does not exist.",
        },
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Evaluation not found."


# ---------------------------------------------------------
# TEST 5: AUTHENTICATION
# ---------------------------------------------------------

def test_get_evaluation_requires_authentication():
    response = client.get(
        "/api/evaluation/eval-test-001"
    )

    assert response.status_code == 401


def test_submit_human_review_requires_authentication():
    response = client.post(
        "/api/evaluation/eval-test-001/review",
        json={
            "decision": "HUMAN_REVIEW",
            "feedback_category": "incorrect_answer",
            "feedback_text": "Unauthorized review attempt.",
        },
    )

    assert response.status_code == 401


# ---------------------------------------------------------
# TEST 6: USER ISOLATION
# ---------------------------------------------------------

def test_user_cannot_access_another_users_evaluation(monkeypatch):
    """Verify evaluation access is scoped to the authenticated user."""

    owner_email = "evaluation_owner_test@example.com"
    attacker_email = "evaluation_attacker_test@example.com"

    get_test_user_token(owner_email)
    attacker_headers = auth_headers_for(attacker_email)

    mock_evaluation = build_mock_evaluation()
    calls = []

    def mock_load_evaluation(evaluation_id, user_id=None):
        calls.append(
            {
                "evaluation_id": evaluation_id,
                "user_id": user_id,
            }
        )

        if evaluation_id == "eval-owned-by-user":
            owner_id = get_test_user_id(owner_email)

            if user_id == owner_id:
                return mock_evaluation

        return None

    monkeypatch.setattr(
        query_route,
        "load_evaluation",
        mock_load_evaluation,
    )

    response = client.get(
        "/api/evaluation/eval-owned-by-user",
        headers=attacker_headers,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Evaluation not found."

    assert calls
    assert calls[-1]["user_id"] == get_test_user_id(
        attacker_email
    )


def test_user_cannot_review_another_users_evaluation(monkeypatch):
    """Verify a user cannot review another user's evaluation."""

    owner_email = "review_owner_test@example.com"
    attacker_email = "review_attacker_test@example.com"

    get_test_user_token(owner_email)

    calls = []

    def mock_load_evaluation(evaluation_id, user_id=None):
        calls.append(
            {
                "evaluation_id": evaluation_id,
                "user_id": user_id,
            }
        )

        return None

    monkeypatch.setattr(
        query_route,
        "load_evaluation",
        mock_load_evaluation,
    )

    response = client.post(
        "/api/evaluation/eval-owned-by-user/review",
        headers=auth_headers_for(attacker_email),
        json={
            "decision": "HUMAN_REVIEW",
            "feedback_category": "incorrect_answer",
            "feedback_text": "Unauthorized review attempt.",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Evaluation not found."

    assert calls
    assert calls[-1]["user_id"] == get_test_user_id(
        attacker_email
    )


# ---------------------------------------------------------
# TEST 7: REVIEW RECOMMENDED DECISION
# ---------------------------------------------------------

def test_ask_route_marks_review_recommended(monkeypatch):
    allow_semantic_security(monkeypatch)

    monkeypatch.setattr(
        query_route,
        "resolve_authorized_api_key",
        lambda user, access_password, user_openai_api_key: "test-api-key",
    )

    monkeypatch.setattr(
        query_route,
        "retrieve_relevant_chunks",
        lambda question, user_id=None: build_mock_retrieval_output(),
    )

    monkeypatch.setattr(
        query_route,
        "get_active_document",
        lambda user_id=None: {"active_doc_id": "test-doc-001"},
    )

    monkeypatch.setattr(
        query_route,
        "generate_grounded_answer",
        lambda question, retrieved_chunks, api_key: (
            "The system uses hybrid retrieval."
        ),
    )

    evaluation = build_mock_evaluation()
    evaluation.decision = EvaluationDecision.REVIEW_RECOMMENDED

    monkeypatch.setattr(
        query_route,
        "evaluate_rag_response",
        lambda **kwargs: evaluation,
    )

    saved_evaluations = mock_evaluation_storage(monkeypatch)

    response = client.post(
        "/api/ask",
        headers=auth_headers(),
        json={
            "question": "How does the system retrieve evidence?",
            "access_password": "test-password",
            "user_openai_api_key": "",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "review_recommended"
    assert data["answer"] == "The system uses hybrid retrieval."
    assert data["evaluation"]["decision"] == "REVIEW_RECOMMENDED"
    assert data["evaluation"]["evaluation_id"] == "eval-test-001"

    assert_evaluation_saved(
        saved_evaluations,
        evaluation,
    )


# ---------------------------------------------------------
# TEST 8: HUMAN REVIEW REQUIRED
# ---------------------------------------------------------

def test_ask_route_withholds_human_review_answer(monkeypatch):
    allow_semantic_security(monkeypatch)

    monkeypatch.setattr(
        query_route,
        "resolve_authorized_api_key",
        lambda user, access_password, user_openai_api_key: "test-api-key",
    )

    monkeypatch.setattr(
        query_route,
        "retrieve_relevant_chunks",
        lambda question, user_id=None: build_mock_retrieval_output(),
    )

    monkeypatch.setattr(
        query_route,
        "get_active_document",
        lambda user_id=None: {"active_doc_id": "test-doc-001"},
    )

    monkeypatch.setattr(
        query_route,
        "generate_grounded_answer",
        lambda question, retrieved_chunks, api_key: (
            "Potentially unsupported answer."
        ),
    )

    evaluation = build_mock_evaluation()
    evaluation.decision = EvaluationDecision.HUMAN_REVIEW

    monkeypatch.setattr(
        query_route,
        "evaluate_rag_response",
        lambda **kwargs: evaluation,
    )

    saved_evaluations = mock_evaluation_storage(monkeypatch)

    response = client.post(
        "/api/ask",
        headers=auth_headers(),
        json={
            "question": "How does the system retrieve evidence?",
            "access_password": "test-password",
            "user_openai_api_key": "",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "human_review_required"
    assert data["answer"] is None
    assert data["evaluation"]["decision"] == "HUMAN_REVIEW"
    assert data["evaluation"]["evaluation_id"] == "eval-test-001"

    assert_evaluation_saved(
        saved_evaluations,
        evaluation,
    )
