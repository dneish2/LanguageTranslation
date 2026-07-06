"""
Server-side Supabase JWT verification (no network, no real key — a throwaway test secret).
Ported from finplatform's tests/test_auth_jwt.py (2026-07-06) — the verification logic and
its test surface are generic, not finplatform-specific. Passage has no Role/credits concept
yet, so the identity model here is just (user_id, email) or (None, None) for anonymous.

Locks in the auth gate's guarantees: a valid token verifies and yields the right identity; a
tampered, expired, wrong-audience, or wrong-secret token is rejected (-> anonymous, never
trusted); and with no secret configured everything is anonymous.
"""
import base64
import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from passage.auth.jwt_verify import (
    verify_supabase_jwt, claims_to_identity, identity_from_auth_header)

SECRET = "test-jwt-secret-do-not-use-in-prod"


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _sign(claims: dict, secret: str = SECRET, alg: str = "HS256") -> str:
    h = _b64(json.dumps({"alg": alg, "typ": "JWT"}).encode())
    p = _b64(json.dumps(claims).encode())
    sig = _b64(hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest())
    return f"{h}.{p}.{sig}"


def _claims(**over):
    base = {"sub": "user-123", "email": "a@b.com", "aud": "authenticated",
            "exp": int(time.time()) + 3600}
    base.update(over)
    return base


def test_valid_token_verifies_and_extracts_identity():
    c = verify_supabase_jwt(_sign(_claims()), SECRET)
    assert c is not None and c["sub"] == "user-123"
    uid, email = claims_to_identity(c)
    assert uid == "user-123" and email == "a@b.com"


def test_tampered_signature_rejected():
    tok = _sign(_claims())
    bad = tok[:-3] + ("aaa" if not tok.endswith("aaa") else "bbb")
    assert verify_supabase_jwt(bad, SECRET) is None


def test_wrong_secret_rejected():
    assert verify_supabase_jwt(_sign(_claims()), "a-different-secret") is None


def test_expired_token_rejected():
    assert verify_supabase_jwt(_sign(_claims(exp=int(time.time()) - 10000)), SECRET) is None


def test_wrong_audience_rejected():
    assert verify_supabase_jwt(_sign(_claims(aud="some-other-aud")), SECRET) is None


def test_garbage_and_empty_rejected():
    for bad in ("", "not.a.jwt", "a.b", None):
        assert verify_supabase_jwt(bad, SECRET) is None


def test_claims_to_identity_defaults_to_anonymous():
    assert claims_to_identity(None) == (None, None)
    assert claims_to_identity({}) == (None, None)


def test_claims_to_identity_lowercases_email():
    uid, email = claims_to_identity({"sub": "x", "email": "MiXeD@Example.com"})
    assert (uid, email) == ("x", "mixed@example.com")


def test_identity_from_header_requires_configured_secret():
    saved = os.environ.pop("SUPABASE_JWT_SECRET", None)
    try:
        # No secret configured -> always anonymous, even with a real-looking token.
        assert identity_from_auth_header(f"Bearer {_sign(_claims())}") == (None, None)
        os.environ["SUPABASE_JWT_SECRET"] = SECRET
        uid, email = identity_from_auth_header(f"Bearer {_sign(_claims())}")
        assert uid == "user-123" and email == "a@b.com"
        assert identity_from_auth_header(None) == (None, None)
        assert identity_from_auth_header("Bearer garbage.token.here") == (None, None)
    finally:
        os.environ.pop("SUPABASE_JWT_SECRET", None)
        if saved is not None:
            os.environ["SUPABASE_JWT_SECRET"] = saved


def test_identity_from_header_accepts_raw_token_without_bearer_prefix():
    os.environ["SUPABASE_JWT_SECRET"] = SECRET
    try:
        uid, email = identity_from_auth_header(_sign(_claims()))
        assert uid == "user-123" and email == "a@b.com"
    finally:
        os.environ.pop("SUPABASE_JWT_SECRET", None)


def test_unsupported_algorithm_rejected():
    tok = _sign(_claims(), alg="none")
    assert verify_supabase_jwt(tok, SECRET) is None


def test_asymmetric_path_returns_none_without_supabase_url():
    saved = os.environ.pop("SUPABASE_URL", None)
    try:
        # ES256 header, no SUPABASE_URL configured -> can't fetch JWKS -> None, not a crash.
        header = _b64(json.dumps({"alg": "ES256", "typ": "JWT"}).encode())
        payload = _b64(json.dumps(_claims()).encode())
        fake_token = f"{header}.{payload}.fakesig"
        assert verify_supabase_jwt(fake_token) is None
    finally:
        if saved is not None:
            os.environ["SUPABASE_URL"] = saved
