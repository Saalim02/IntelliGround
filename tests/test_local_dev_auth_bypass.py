from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services.security import auth_dependencies


def test_local_dev_identity_is_available_without_session(monkeypatch):
    monkeypatch.setattr(auth_dependencies, "LOCAL_DEV_AUTH_BYPASS", True)

    response = TestClient(app).get("/api/auth/me")

    assert response.status_code == 200
    assert response.json()["email"] == "localdev@intelliground.local"


def test_page_images_use_local_identity_without_session(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_dependencies, "LOCAL_DEV_AUTH_BYPASS", True)

    page_dir = tmp_path / "page-images"
    document_dir = page_dir / "local-document"
    document_dir.mkdir(parents=True)
    (document_dir / "page_3.png").write_bytes(b"png-test")
    monkeypatch.setattr(
        "app.api.routes.page_images.get_user_page_image_dir",
        lambda _user_id: Path(page_dir),
    )

    response = TestClient(app).get(
        "/api/page-images/local-document/page_3.png"
    )

    assert response.status_code == 200
    assert response.content == b"png-test"
