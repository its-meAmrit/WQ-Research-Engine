"""What can go wrong with a web service, each said in words a person can act on.

None of these ever carries a request, a header or a key: messages are built from the provider's
name and the HTTP status alone.
"""


class IntegrationError(Exception):
    """Base of every integration failure.

    ``code`` is stable and machine-readable; ``http_status`` is what the API answers when one
    escapes a route; ``retryable`` says whether trying again later can help.
    """

    code = "integration_error"
    http_status = 400
    retryable = False

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# --- the service answered (or did not) ---------------------------------------------------


class IntegrationAuthError(IntegrationError):
    """The provider refused the key."""

    code = "auth_failed"
    http_status = 502


class IntegrationQuotaError(IntegrationError):
    """The plan, key or pay-as-you-go limit is used up. Waiting a minute will not help."""

    code = "quota_exceeded"
    http_status = 502


class IntegrationRateLimited(IntegrationError):
    code = "rate_limited"
    http_status = 429
    retryable = True


class IntegrationTimeout(IntegrationError):
    code = "timeout"
    http_status = 504
    retryable = True


class IntegrationUnavailable(IntegrationError):
    """The provider failed on its side."""

    code = "unavailable"
    http_status = 502
    retryable = True


class IntegrationNetworkError(IntegrationError):
    """The provider could not be reached at all."""

    code = "network"
    http_status = 502
    retryable = True


class IntegrationBadResponse(IntegrationError):
    code = "bad_response"
    http_status = 502


class IntegrationRejected(IntegrationError):
    """The provider understood the request and refused it: a limit, a malformed field."""

    code = "rejected"
    http_status = 422


# --- the request to *this* application ---------------------------------------------------


class RequestBudgetExhaustedError(IntegrationError):
    """The user's own request budget for today is spent."""

    code = "budget_exhausted"
    http_status = 429


class UnknownProviderError(IntegrationError):
    code = "unknown_provider"
    http_status = 404


class NotConfiguredError(IntegrationError):
    code = "not_configured"
    http_status = 404


class InvalidKeyError(IntegrationError):
    """The key is malformed: empty, with whitespace inside it, or implausibly long."""

    code = "invalid_key"
    http_status = 422


class DuplicateKeyError(IntegrationError):
    code = "duplicate_key"
    http_status = 409


class ConfirmationRequiredError(IntegrationError):
    """The action may spend a paid credit and the user has not said that is fine."""

    code = "confirmation_required"
    http_status = 409


class UnsupportedCapabilityError(IntegrationError):
    code = "unsupported_capability"
    http_status = 422


class UnreadableKeyError(IntegrationError):
    """A stored key could not be decrypted: the vault key changed or the row was altered."""

    code = "key_unreadable"
    http_status = 409
