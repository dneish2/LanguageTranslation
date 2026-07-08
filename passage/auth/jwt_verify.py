"""
passage.auth.jwt_verify
========================
Verify a Supabase access token (a JWT) on the server. Ported from
finplatform's `finplatform/auth/jwt_verify.py` (2026-07-06) — the
verification logic is generic, not finplatform-specific, and Phase 4's
locked decision is to share that Supabase project. Only the identity
model was simplified: finplatform maps claims to a paid-tier Role;
Passage (no billing yet) just needs (uid, email) or anonymous.

Supports BOTH signing schemes Supabase uses:

  * HS256 (legacy) — symmetric, verified against the shared `SUPABASE_JWT_SECRET` using ONLY the stdlib.
  * ES256 / RS256 (current) — asymmetric. Supabase migrated projects to per-project JWT *signing keys*;
    new user access tokens are signed with a private key and verified against the project's PUBLIC key,
    published at `${SUPABASE_URL}/auth/v1/.well-known/jwks.json`. We fetch that JWKS (cached) and verify
    via PyJWT — the server never holds a signing secret, which is strictly more secure.

The token's header `alg` selects the path, so a project on either scheme (or mid-rotation, serving both)
works. PyJWT is imported LAZILY only for the asymmetric path, so the stdlib HS256 path — and the offline
test suite — need no third-party dependency.

Posture: anything that doesn't verify returns None, and the caller treats None as ANONYMOUS (still gated)
rather than a hard error — a missing/expired/garbage token degrades to the anon experience, never a 500.
The server is the authority (the frontend role is display-only).

Zero-config by design: with no SUPABASE_URL/SUPABASE_JWT_SECRET set, every token is unverifiable and every
caller resolves to anonymous — Passage keeps working fully signed-out until David configures real values.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time


def _b64url_decode(seg: str) -> bytes:
    return base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))


def _check_claims(claims, audience: str, leeway: int):
    """Shared exp / nbf / aud validation. Returns the claims dict, or None if any check fails."""
    if not isinstance(claims, dict):
        return None
    now = time.time()
    exp = claims.get("exp")
    if isinstance(exp, (int, float)) and now > exp + leeway:
        return None
    nbf = claims.get("nbf")
    if isinstance(nbf, (int, float)) and now + leeway < nbf:
        return None
    if audience:
        aud = claims.get("aud")
        auds = aud if isinstance(aud, list) else [aud]
        if audience not in auds:
            return None
    return claims


def _verify_hs256(header_b64, payload_b64, sig_b64, secret, audience, leeway):
    """HS256 with the shared secret (arg or `SUPABASE_JWT_SECRET`), stdlib only."""
    secret = secret or os.environ.get("SUPABASE_JWT_SECRET")
    if not secret:
        return None
    try:
        signing_input = f"{header_b64}.{payload_b64}".encode()
        expected = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64url_decode(sig_b64)):
            return None
        claims = json.loads(_b64url_decode(payload_b64))
    except Exception:
        return None
    return _check_claims(claims, audience, leeway)


_JWKS_CLIENTS: dict = {}   # jwks_url -> PyJWKClient (caches keys, so we don't refetch per request)


def _verify_asymmetric(token, alg, audience, leeway):
    """ES256/RS256 against the project's published JWKS (needs `SUPABASE_URL` + PyJWT[crypto]). Returns
    None on any failure (unreachable JWKS, unknown key, bad signature, PyJWT not installed)."""
    base = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
    if not base:
        return None
    try:
        import jwt  # PyJWT[crypto]; lazy so the stdlib HS256 path never needs it
        url = base + "/auth/v1/.well-known/jwks.json"
        client = _JWKS_CLIENTS.get(url)
        if client is None:
            from jwt import PyJWKClient
            client = PyJWKClient(url)
            _JWKS_CLIENTS[url] = client
        signing_key = client.get_signing_key_from_jwt(token)
        claims = jwt.decode(token, signing_key.key, algorithms=[alg],
                            audience=(audience or None), leeway=leeway)
        return claims if isinstance(claims, dict) else None
    except Exception:
        return None


def verify_supabase_jwt(token, secret: str | None = None, *, audience: str = "authenticated", leeway: int = 60):
    """Return the verified claims dict, or None. Routes on the token's `alg`: HS256 -> shared secret (stdlib),
    ES256/RS256 -> the project's JWKS (PyJWT). Validates signature + exp/nbf + aud in every path."""
    if not token or not isinstance(token, str):
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None
    header_b64, payload_b64, sig_b64 = parts
    try:
        alg = json.loads(_b64url_decode(header_b64)).get("alg")
    except Exception:
        return None
    if alg == "HS256":
        return _verify_hs256(header_b64, payload_b64, sig_b64, secret, audience, leeway)
    if alg in ("ES256", "RS256"):
        return _verify_asymmetric(token, alg, audience, leeway)
    return None   # unknown/unsupported alg (incl. "none") -> reject


def claims_to_identity(claims):
    """(user_id, email) from verified claims, or (None, None) if unverified/anonymous.

    No Role/credits concept yet — Passage has no billing. Any signed-in
    identity is just "signed in"; per-user data scoping (Phase 4 item 3)
    keys off `user_id`, not a tier.
    """
    if not claims:
        return None, None
    uid = claims.get("sub")
    email = (claims.get("email") or "").lower() or None
    return uid, email


def identity_from_auth_header(auth_header):
    """Resolve (user_id, email) from an HTTP `Authorization` header. Verifies HS256 (shared secret)
    OR ES256/RS256 (JWKS) depending on the token. No header / invalid token -> (None, None). Never raises."""
    if not auth_header:
        return None, None
    parts = auth_header.split()
    token = parts[1] if (len(parts) == 2 and parts[0].lower() == "bearer") else auth_header
    return claims_to_identity(verify_supabase_jwt(token))
