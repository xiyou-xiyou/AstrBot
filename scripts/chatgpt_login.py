"""Authorize AstrBot to use a ChatGPT subscription through official OAuth."""

import base64
import hashlib
import html
import json
import os
import secrets
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import jwt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astrbot.core.utils.astrbot_path import get_astrbot_data_path  # noqa: E402


def main() -> None:
    """Run a loopback login listener and validate credentials before saving them.

    Raises:
        RuntimeError: If authorization, identity validation, or scopes fail.
    """
    directory = Path(get_astrbot_data_path()) / "chatgpt"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    host_path = directory / "host_id"
    if not host_path.exists():
        host_path.write_text(f"urn:uuid:{uuid.uuid4()}", encoding="utf-8")
    host_id = host_path.read_text(encoding="utf-8").strip()
    credential_path = directory / "credentials.json"
    saved = (
        json.loads(credential_path.read_text(encoding="utf-8"))
        if credential_path.exists()
        else {}
    )
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    callback: dict[str, str] = {}
    authorization_url = ""

    class Handler(BaseHTTPRequestHandler):
        """Serve the login link and accept only the matching OAuth callback."""

        def log_message(self, format: str, *args) -> None:
            """Suppress HTTP logs to avoid logging authorization codes.

            Args:
                format: HTTP log format.
                args: HTTP log arguments.
            """

        def do_GET(self) -> None:
            """Handle a login-page request or a validated callback."""
            url = urlsplit(self.path)
            status = 200
            if url.path == "/":
                body = (
                    "<h1>AstrBot</h1><p>Authorize access to your ChatGPT plan.</p>"
                    f'<a href="{html.escape(authorization_url, quote=True)}">'
                    "Continue with ChatGPT</a>"
                )
            elif url.path == "/auth/callback":
                values = parse_qs(url.query)
                returned_state = values.get("state", [""])[0]
                if not secrets.compare_digest(returned_state, state):
                    status, body = 400, "Invalid authorization state."
                elif callback:
                    status, body = 400, "Authorization already received."
                else:
                    callback.update({k: v[0] for k, v in values.items()})
                    body = "Authorization received. AstrBot is validating the result."
            else:
                status, body = 404, "Not found."
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body.encode())

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        server.timeout = 1
        redirect_uri = f"http://127.0.0.1:{server.server_port}/auth/callback"
        parameters = {
            "client_id": saved.get("client_id", "dynamic_agent_client"),
            "ext_agent_host_id": host_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct",
            "resource": "https://api.openai.com/v1",
            "state": state,
            "nonce": nonce,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
        }
        if saved:
            parameters["id_token_hint"] = saved["id_token"]
        else:
            parameters["agent_name_hint"] = "AstrBot"
        authorization_url = (
            "https://auth.openai.com/api/accounts/authorize?" + urlencode(parameters)
        )
        print(
            f"Continue with ChatGPT: http://127.0.0.1:{server.server_port}/", flush=True
        )
        deadline = time.monotonic() + 600
        while not callback and time.monotonic() < deadline:
            server.handle_request()
    if not callback:
        raise RuntimeError("Login timed out. Run the login command again.")
    if callback.get("error"):
        raise RuntimeError("Authorization was denied or failed. No credentials saved.")
    client_id = callback.get("client_id", saved.get("client_id"))
    if not client_id or client_id == "dynamic_agent_client":
        raise RuntimeError("Authorization did not return an issued client ID.")
    if saved and client_id != saved["client_id"]:
        raise RuntimeError("The returned client ID does not match this registration.")
    with httpx.Client(timeout=30) as client:
        response = client.post(
            "https://auth.openai.com/api/accounts/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": callback["code"],
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "resource": "https://api.openai.com/v1",
            },
        )
        if response.status_code != 200:
            raise RuntimeError(f"Token exchange failed (HTTP {response.status_code}).")
        tokens = response.json()
        scopes = tokens.get("scope", "").split()
        if "chatgpt.tokens.use.direct" not in scopes:
            raise RuntimeError("ChatGPT plan usage permission was not granted.")
        keys_response = client.get("https://auth.openai.com/.well-known/jwks.json")
        keys_response.raise_for_status()
        header = jwt.get_unverified_header(tokens["id_token"])
        key = next(k for k in keys_response.json()["keys"] if k["kid"] == header["kid"])
        identity = jwt.decode(
            tokens["id_token"],
            jwt.PyJWK.from_dict(key).key,
            algorithms=["RS256"],
            audience=client_id,
            issuer="https://auth.openai.com",
            options={"require": ["sub", "exp", "iat", "nonce"]},
        )
        if identity["nonce"] != nonce:
            raise RuntimeError("The identity token nonce does not match.")
        if saved and identity["sub"] != saved["subject"]:
            raise RuntimeError(
                "The signed-in account does not match the saved account."
            )
    record = {
        **tokens,
        "client_id": client_id,
        "subject": identity["sub"],
        "ext_agent_host_id": host_id,
        "scopes": scopes,
        "expires_at": time.time() + tokens["expires_in"],
    }
    temporary = credential_path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(record, output)
    temporary.replace(credential_path)
    print("ChatGPT subscription authorization saved successfully.", flush=True)


if __name__ == "__main__":
    main()
