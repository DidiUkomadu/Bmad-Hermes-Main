"""Password hashing — Epic 8 (change proposal D2).

scrypt from the standard library, a random 16-byte salt per password, and a
self-describing encoded form so parameters can be raised later:

    scrypt$<n>$<r>$<p>$<salt b64>$<hash b64>
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

_N, _R, _P = 2**14, 8, 1
_KEY_LENGTH = 32
_MAX_PASSWORD_BYTES = 1024  # bounds hashing work for absurd inputs


def hash_password(password: str) -> str:
    """Return an encoded scrypt hash of *password*."""
    salt = secrets.token_bytes(16)
    digest = _scrypt(password, salt, _N, _R, _P)
    return "$".join(["scrypt", str(_N), str(_R), str(_P), _b64(salt), _b64(digest)])


def verify_password(password: str, encoded: str) -> bool:
    """Check *password* against an encoded hash in constant time."""
    try:
        scheme, n, r, p, salt, expected = encoded.split("$")
        if scheme != "scrypt":
            return False
        digest = _scrypt(password, _unb64(salt), int(n), int(r), int(p))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, _unb64(expected))


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    data = password.encode("utf-8")[:_MAX_PASSWORD_BYTES]
    return hashlib.scrypt(data, salt=salt, n=n, r=r, p=p, dklen=_KEY_LENGTH, maxmem=2**26)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
