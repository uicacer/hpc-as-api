"""
STREAM Proxy — Authentication & Authorization

This module handles two authentication modes that can coexist on the same proxy:

MODE A: Globus Token Auth (preferred for university-wide deployment)
----------------------------------------------------------------------
The caller presents a Globus access token in the Authorization header.
The proxy validates it against Globus Auth's introspect endpoint and extracts
the caller's identity (e.g., nassar@uic.edu). The identity is used for:
  - Access control (domain check, e.g. @uic.edu only)
  - Per-request attribution in proxy logs
  - Rate limiting per individual user identity

NOTE: The proxy currently submits all Globus Compute jobs under its own stored
credentials (~/.globus_compute/storage.db), not under the caller's token.
Wiring caller.globus_token through to globus_compute_sdk for true per-user
SLURM attribution is a planned extension.

MODE B: API Key Auth (for external services like AWS/Amplify)
--------------------------------------------------------------
The caller presents a pre-issued service key (e.g., "sk-stream-amplify").
The proxy validates it against a local key table and logs the service name.
Used when the caller is a server (not a human) that authenticates its own users
separately (e.g., AWS Cognito). Per-user attribution lives in the caller's own
logs, not on Lakeshore.

The @uic.edu problem with Amplify users:
-----------------------------------------
If an Amplify user logs in with AWS Cognito, the proxy sees the Amplify server's
service key — not the user's UIC identity. There is no way for the proxy or
Lakeshore to know who that end user is unless Amplify implements one of:

  Option 1 (recommended): Amplify adds "Login with Globus" as an auth option.
    The user links their UIC/Globus account during Amplify signup. Amplify then
    holds a Globus token for that user and passes it to the proxy. Full per-user
    attribution on Lakeshore, no changes needed on the proxy side.

  Option 2 (simpler): Amplify authenticates as a service (Mode B above).
    Per-user attribution lives in Amplify's own logs. Acceptable for an
    institutional service operated by a trusted team.

AUTH HEADER FORMAT:
  Globus token:  Authorization: Bearer <globus_access_token>
  API key:       Authorization: Bearer sk-stream-<key>

The proxy distinguishes them by attempting Globus introspection first. If the
token is not a valid Globus token, it falls back to API key validation.
"""

import hashlib
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import httpx
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# =============================================================================
# GLOBUS AUTH CONSTANT
# =============================================================================

GLOBUS_INTROSPECT_URL = "https://auth.globus.org/v2/oauth2/token/introspect"


# =============================================================================
# AUTH CONFIGURATION
# =============================================================================


@dataclass
class AuthConfig:
    """
    All authentication settings for the proxy, in one place.

    Every field falls back to its corresponding environment variable when not
    supplied — so existing env-var-based deployments work without any changes.
    Pass an ``AuthConfig`` instance to ``HPCApp`` or ``create_openai_app`` to
    configure auth entirely in Python, with no ``.env`` files needed.

    **Rate limiting** is per-caller over a sliding window.  Two levels:

    * ``rate_limit_requests`` / ``PROXY_RATE_LIMIT_REQUESTS`` — global default
      applied to every caller that doesn't have a per-key override.
    * ``per_key_rate_limits`` / ``PROXY_RATE_LIMIT_REQUESTS_<NAME>`` — override
      for a specific named key.  ``<NAME>`` is the suffix used in
      ``PROXY_API_KEY_<NAME>``.  Example: if you defined
      ``PROXY_API_KEY_AMPLIFY=sk-...``, set
      ``PROXY_RATE_LIMIT_REQUESTS_AMPLIFY=200`` to give that key 200 req/min.

    Example — per-key limits::

        auth = AuthConfig(
            api_keys={"amplify": "sk-amplify-key", "demo": "sk-demo-key"},
            rate_limit_requests=10000,   # default (catch runaway scripts)
            rate_limit_window=60,
            per_key_rate_limits={"amplify": 500, "demo": 20},
        )

    +-----------------------------------+------------------------------------------+
    | Argument                          | Env var fallback                         |
    +===================================+==========================================+
    | ``globus_client_id``              | ``GLOBUS_CLIENT_ID``                     |
    | ``globus_client_secret``          | ``GLOBUS_CLIENT_SECRET``                 |
    | ``allowed_domains``               | ``PROXY_ALLOWED_DOMAINS``                |
    | ``api_keys``                      | ``PROXY_API_KEY_<NAME>``                 |
    | ``rate_limit_requests``           | ``PROXY_RATE_LIMIT_REQUESTS``            |
    | ``rate_limit_window``             | ``PROXY_RATE_LIMIT_WINDOW``              |
    | ``per_key_rate_limits``           | ``PROXY_RATE_LIMIT_REQUESTS_<NAME>``     |
    +-----------------------------------+------------------------------------------+
    """

    globus_client_id: str = field(default_factory=lambda: os.getenv("GLOBUS_CLIENT_ID", ""))
    globus_client_secret: str = field(default_factory=lambda: os.getenv("GLOBUS_CLIENT_SECRET", ""))

    # Allowed Globus email domains. Empty list = accept any valid Globus identity.
    # Example: ["uic.edu", "anl.gov"] restricts to those institutions.
    allowed_domains: list[str] = field(
        default_factory=lambda: [d.strip() for d in os.getenv("PROXY_ALLOWED_DOMAINS", "").split(",") if d.strip()]
    )

    # API keys for service-to-service callers. Keys are mapped to service names
    # for logging. Populated from PROXY_API_KEY_<NAME> env vars by default.
    api_keys: dict[str, str] = field(default_factory=lambda: _load_api_keys_from_env())

    # Global default rate limit (requests per window). Intentionally high so
    # shared classroom keys aren't throttled — vLLM handles backpressure internally.
    # Lower only to catch runaway scripts, not to shape normal classroom load.
    rate_limit_requests: int = field(default_factory=lambda: int(os.getenv("PROXY_RATE_LIMIT_REQUESTS", "10000")))
    rate_limit_window: int = field(default_factory=lambda: int(os.getenv("PROXY_RATE_LIMIT_WINDOW", "60")))

    # Per-key overrides: service_name → max requests per window.
    # Populated from PROXY_RATE_LIMIT_REQUESTS_<NAME> env vars by default.
    # Takes precedence over rate_limit_requests for that specific caller.
    per_key_rate_limits: dict[str, int] = field(default_factory=lambda: _load_per_key_limits_from_env())


def _load_api_keys_from_env() -> dict[str, str]:
    """Load API keys from PROXY_API_KEY_<NAME> env vars. key → service_name."""
    table: dict[str, str] = {}
    for name, val in os.environ.items():
        if name.startswith("PROXY_API_KEY_") and val:
            table[val] = name[len("PROXY_API_KEY_") :].lower()
    legacy = os.getenv("PROXY_API_KEY", "")
    if legacy and legacy not in table:
        table[legacy] = "legacy"
    return table


def _load_per_key_limits_from_env() -> dict[str, int]:
    """Load per-key rate limits from PROXY_RATE_LIMIT_REQUESTS_<NAME> env vars.

    Returns service_name → max_requests mapping.  Names are lowercased to match
    the service names produced by _load_api_keys_from_env().
    """
    limits: dict[str, int] = {}
    prefix = "PROXY_RATE_LIMIT_REQUESTS_"
    for name, val in os.environ.items():
        if name.startswith(prefix) and val:
            service = name[len(prefix) :].lower()
            try:
                limits[service] = int(val)
            except ValueError:
                logger.warning(f"Ignoring invalid rate limit for {service}: {val!r}")
    return limits


# Module-level defaults (used by the legacy module-level `authenticate` function).
# New code should use Authenticator(AuthConfig()) instead.
GLOBUS_CLIENT_ID = os.getenv("GLOBUS_CLIENT_ID", "")
GLOBUS_CLIENT_SECRET = os.getenv("GLOBUS_CLIENT_SECRET", "")  # pragma: allowlist secret
ALLOWED_DOMAINS = [d.strip() for d in os.getenv("PROXY_ALLOWED_DOMAINS", "").split(",") if d.strip()]
_RAW_API_KEY_TABLE: dict[str, str] = _load_api_keys_from_env()
RATE_LIMIT_REQUESTS = int(os.getenv("PROXY_RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("PROXY_RATE_LIMIT_WINDOW", "60"))

# In-memory store: caller_id → list of request timestamps in the current window
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(caller_id: str) -> None:
    """
    Enforce per-caller rate limit. Raises HTTP 429 if the caller exceeds
    RATE_LIMIT_REQUESTS requests within RATE_LIMIT_WINDOW_SECONDS seconds.

    Uses a sliding window — only counts requests within the last N seconds,
    so the limit resets naturally without a cron job.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    # Remove timestamps outside the current window (sliding window cleanup)
    timestamps = _rate_limit_store[caller_id]
    _rate_limit_store[caller_id] = [t for t in timestamps if t > window_start]

    if len(_rate_limit_store[caller_id]) >= RATE_LIMIT_REQUESTS:
        logger.warning(
            f"Rate limit exceeded: caller={caller_id}, "
            f"requests={len(_rate_limit_store[caller_id])}/{RATE_LIMIT_REQUESTS} "
            f"in {RATE_LIMIT_WINDOW_SECONDS}s window"
        )
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per "
                f"{RATE_LIMIT_WINDOW_SECONDS}s. Please slow down."
            ),
        )

    _rate_limit_store[caller_id].append(now)


# =============================================================================
# CALLER IDENTITY
# =============================================================================


@dataclass
class CallerIdentity:
    """Represents an authenticated caller — either a Globus user or an API-key service."""

    name: str
    auth_mode: str
    globus_token: str | None = None
    credential_hash: str = field(default="")

    def log_safe_id(self) -> str:
        return f"{self.auth_mode}:{self.name}:{self.credential_hash[:8]}"


# =============================================================================
# AUTHENTICATOR — stateful, config-driven auth dependency
# =============================================================================


class Authenticator:
    """
    FastAPI-compatible authentication dependency, fully configurable in Python.

    Pass an instance to ``HPCApp`` or ``create_openai_app`` via the ``auth``
    parameter to configure auth without environment variables::

        from hpc_as_api.auth import AuthConfig, Authenticator

        auth = Authenticator(AuthConfig(
            globus_client_id="...",
            globus_client_secret="...",
            allowed_domains=["ornl.gov"],
            api_keys={"myservice": "sk-my-key"},
        ))

        gateway = HPCApp(endpoint_id="...", relay_url="wss://...", auth=auth)

    The ``auth`` parameter on ``HPCApp``/``create_openai_app`` also accepts a
    plain ``AuthConfig`` and wraps it automatically.
    """

    def __init__(self, config: "AuthConfig | None" = None):
        self.config = config or AuthConfig()
        self._security = HTTPBearer()
        self._rate_store: dict[str, list[float]] = defaultdict(list)

    async def _validate_globus(self, token: str) -> CallerIdentity | None:
        cfg = self.config
        if not cfg.globus_client_id or not cfg.globus_client_secret:
            return None
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    GLOBUS_INTROSPECT_URL,
                    auth=(cfg.globus_client_id, cfg.globus_client_secret),
                    data={"token": token, "include": "identities_set"},
                )
            if resp.status_code != 200:
                return None
            info = resp.json()
            if not info.get("active", False):
                return None
            email = info.get("email", "") or info.get("username", "")
            if not email:
                return None
            if cfg.allowed_domains:
                domain = email.split("@")[-1].lower() if "@" in email else ""
                if domain not in [d.lower() for d in cfg.allowed_domains]:
                    raise HTTPException(
                        status_code=403,
                        detail=(
                            f"Access restricted to: {', '.join(cfg.allowed_domains)}. "
                            f"Your identity ({email}) is not from an allowed institution."
                        ),
                    )
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            logger.info(f"Globus token validated: identity={email}")
            return CallerIdentity(name=email, auth_mode="globus", globus_token=token, credential_hash=token_hash)
        except HTTPException:
            raise
        except Exception as e:
            logger.debug(f"Globus token validation failed: {e}")
            return None

    def _validate_api_key(self, token: str) -> CallerIdentity | None:
        service_name = self.config.api_keys.get(token)
        if not service_name:
            return None
        key_hash = hashlib.sha256(token.encode()).hexdigest()
        logger.info(f"API key validated: service={service_name}, key_hash={key_hash[:16]}")
        return CallerIdentity(name=service_name, auth_mode="api_key", globus_token=None, credential_hash=key_hash)

    def _check_rate_limit(self, caller_id: str) -> None:
        cfg = self.config
        # Per-key override takes precedence over the global default.
        limit = cfg.per_key_rate_limits.get(caller_id, cfg.rate_limit_requests)
        now = time.time()
        window_start = now - cfg.rate_limit_window
        self._rate_store[caller_id] = [t for t in self._rate_store[caller_id] if t > window_start]
        if len(self._rate_store[caller_id]) >= limit:
            raise HTTPException(
                status_code=429,
                detail=(f"Rate limit exceeded: {limit} requests per {cfg.rate_limit_window}s. Please slow down."),
            )
        self._rate_store[caller_id].append(now)

    async def __call__(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    ) -> CallerIdentity:
        token = credentials.credentials
        caller = await self._validate_globus(token)
        if caller is None:
            caller = self._validate_api_key(token)
        if caller is None:
            logger.warning(f"Authentication failed from {request.client.host if request.client else 'unknown'}")
            raise HTTPException(
                status_code=401,
                detail=(
                    "Authentication required. Provide either:\n"
                    "  • A valid Globus access token\n"
                    "  • A pre-issued service API key\n"
                    "Contact your proxy administrator for access."
                ),
            )
        self._check_rate_limit(caller.name)
        logger.info(
            f"Authenticated: caller={caller.log_safe_id()}, path={request.url.path}, "
            f"client={request.client.host if request.client else 'unknown'}"
        )
        return caller


# =============================================================================
# LEGACY MODULE-LEVEL AUTH FUNCTIONS (backward compatible)
# =============================================================================
# These use the module-level globals above. New code should use Authenticator.

_default_authenticator = Authenticator(
    AuthConfig(
        globus_client_id=GLOBUS_CLIENT_ID,
        globus_client_secret=GLOBUS_CLIENT_SECRET,
        allowed_domains=ALLOWED_DOMAINS,
        api_keys=_RAW_API_KEY_TABLE,
        rate_limit_requests=RATE_LIMIT_REQUESTS,
        rate_limit_window=RATE_LIMIT_WINDOW_SECONDS,
    )
)

security = HTTPBearer()


async def _validate_globus_token(token: str) -> CallerIdentity | None:
    return await _default_authenticator._validate_globus(token)


def _validate_api_key(token: str) -> CallerIdentity | None:
    return _default_authenticator._validate_api_key(token)


async def authenticate(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> CallerIdentity:
    """
    FastAPI dependency — authenticates every request to the proxy.

    Uses module-level configuration (env vars). For programmatic config,
    use ``Authenticator(AuthConfig(...))`` instead.
    """
    token = credentials.credentials
    caller = await _validate_globus_token(token)
    if caller is None:
        caller = _validate_api_key(token)
    if caller is None:
        logger.warning(f"Authentication failed from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(
            status_code=401,
            detail=(
                "Authentication required. Provide either:\n"
                "  • A valid Globus access token (for institutional users)\n"
                "  • A pre-issued service API key (for application integrations)\n"
                "Contact your proxy administrator for access."
            ),
        )

    # Step 4: Rate limiting — applied after auth so we can rate-limit per identity
    _check_rate_limit(caller.name)

    # Step 5: Log the request with full attribution
    # Log the service/user name and credential hash — never the raw token/key
    logger.info(
        f"Authenticated request: caller={caller.log_safe_id()}, "
        f"path={request.url.path}, "
        f"client={request.client.host if request.client else 'unknown'}"
    )

    return caller


# =============================================================================
# INPUT VALIDATION
# =============================================================================


def validate_messages(messages: list) -> list:
    """
    Validate and sanitize the messages array before forwarding to Globus Compute.

    Why this matters: the proxy forwards messages to a Globus Compute function
    that runs on the HPC cluster. Malformed payloads could crash vLLM, cause
    unexpected behavior, or consume excessive resources.

    Checks:
      - messages is a list of dicts
      - each message has "role" (one of user/assistant/system) and "content"
      - content is a string or a list (for multimodal messages with images)
      - individual text content is under 100K characters
      - total message count is under 500 turns (prevents context window abuse)

    Returns the validated messages list.
    Raises HTTP 400 with a descriptive message for any violation.
    """
    allowed_roles = {"user", "assistant", "system"}
    max_content_chars = 100_000  # ~75K tokens — well within any model's context
    max_messages = 500  # 250 full back-and-forth turns

    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="'messages' must be a list")

    if len(messages) == 0:
        raise HTTPException(status_code=400, detail="'messages' cannot be empty")

    if len(messages) > max_messages:
        raise HTTPException(status_code=400, detail=f"Too many messages: {len(messages)} > {max_messages} limit")

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise HTTPException(status_code=400, detail=f"Message {i}: must be a dict, got {type(msg).__name__}")

        role = msg.get("role")
        if role not in allowed_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i}: invalid role '{role}'. Must be one of: {allowed_roles}",
            )

        content = msg.get("content")
        if content is None:
            raise HTTPException(status_code=400, detail=f"Message {i}: missing 'content' field")

        # Content can be a string (text only) or a list (multimodal: text + images)
        if isinstance(content, str):
            if len(content) > max_content_chars:
                raise HTTPException(
                    status_code=400,
                    detail=(f"Message {i}: content too large ({len(content):,} chars > {max_content_chars:,} limit)"),
                )
        elif isinstance(content, list):
            # Multimodal content: list of {"type": "text", "text": "..."} or
            # {"type": "image_url", "image_url": {"url": "data:image/..."}}
            # We just check it's a list of dicts — deep validation is handled
            # by vLLM itself when the request arrives on the cluster.
            for j, part in enumerate(content):
                if not isinstance(part, dict):
                    raise HTTPException(status_code=400, detail=f"Message {i}, content part {j}: must be a dict")
        else:
            raise HTTPException(
                status_code=400,
                detail=(f"Message {i}: 'content' must be a string or list, got {type(content).__name__}"),
            )

    return messages
