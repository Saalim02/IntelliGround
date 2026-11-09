import uuid

from fastapi.testclient import TestClient

from app.core.database import SessionLocal
from app.main import app
from app.models.auth_models import User
from app.models.conversation_models import Conversation
from app.models.evaluation_models import EvaluationDecision, EvaluationResult
from app.services.conversation_service import add_turn
from app.services.evaluation.evaluation_storage_service import save_evaluation
from app.services.security.auth_service import create_access_token, create_user


client = TestClient(app)


def build_answer(*, with_evidence=True, with_evaluation=False):
    db = SessionLocal()
    user = create_user(db, f"evaluation-user-{uuid.uuid4().hex}@example.com", "EvaluationPassword123!")
    conversation = Conversation(
        id=str(uuid.uuid4()), user_id=user.id, document_id="doc-owned", document_name="owned.pdf", mode="text", response_language="en"
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    add_turn(db, conversation, transcript="What does the document say?", answer=None, evidence=[], role="user", provider_request_id=str(uuid.uuid4()))
    evidence = [{"page_number": 1, "snippet": "The document says this.", "_retrieved_chunks": [{"chunk_id": 1, "page_number": 1, "text": "The document says this.", "score": 0.1}]}] if with_evidence else []
    answer = add_turn(db, conversation, transcript=None, answer="The document says this.", evidence=evidence, role="assistant", provider_request_id=str(uuid.uuid4()))
    if with_evaluation:
        evaluation = EvaluationResult(
            evaluation_id=f"eval-{uuid.uuid4().hex}", user_id=user.id, question="What does the document say?", answer=answer.answer,
            retrieval_quality=0.9, source_relevance=0.9, grounding=0.9, answer_correctness=0.9, citation_quality=1.0,
            overall_confidence=0.92, decision=EvaluationDecision.AUTO_APPROVE, reasons=["Existing result."], claims=[]
        )
        save_evaluation(evaluation, user_id=user.id)
        answer.evaluation_id = evaluation.evaluation_id
        db.add(answer)
        db.commit()
    conversation_id = conversation.id
    answer_id = answer.id
    token = create_access_token(user.id, user.token_version)
    db.close()
    return conversation_id, answer_id, {"Authorization": f"Bearer {token}"}


def test_user_evaluation_reuses_owned_result():
    conversation_id, answer_id, headers = build_answer(with_evaluation=True)
    response = client.post("/api/evaluation/answer", headers=headers, json={"conversation_id": conversation_id, "answer_id": answer_id})
    assert response.status_code == 200
    assert response.json()["reused"] is True
    assert response.json()["status"] == "completed"


def test_user_evaluation_returns_unavailable_without_pdf_evidence():
    conversation_id, answer_id, headers = build_answer(with_evidence=False)
    response = client.post("/api/evaluation/answer", headers=headers, json={"conversation_id": conversation_id, "answer_id": answer_id})
    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"


def test_user_cannot_evaluate_another_users_answer():
    conversation_id, answer_id, _ = build_answer(with_evaluation=True)
    db = SessionLocal()
    attacker = create_user(db, f"evaluation-attacker-{uuid.uuid4().hex}@example.com", "EvaluationPassword123!")
    headers = {"Authorization": f"Bearer {create_access_token(attacker.id, attacker.token_version)}"}
    db.close()
    response = client.post("/api/evaluation/answer", headers=headers, json={"conversation_id": conversation_id, "answer_id": answer_id})
    assert response.status_code == 404


def test_user_feedback_is_saved_separately():
    conversation_id, answer_id, headers = build_answer(with_evidence=True)
    response = client.post("/api/evaluation/answer/feedback", headers=headers, json={"conversation_id": conversation_id, "answer_id": answer_id, "helpful": False})
    assert response.status_code == 200
    assert response.json() == {"status": "saved", "answer_id": answer_id, "helpful": False}
