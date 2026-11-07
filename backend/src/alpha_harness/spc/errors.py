"""What can go wrong in the SPC lab, each said in words a person can act on."""


class SpcError(Exception):
    """Base of every SPC failure. ``code`` is stable and machine-readable."""

    code = "spc_error"


class SpcNotFoundError(SpcError):
    code = "spc_not_found"


class SpcInvalidError(SpcError):
    """The request is understood and refused: a limit, a range, a missing field."""

    code = "spc_invalid"


class SpcConflictError(SpcError):
    """The request cannot be done in the state things are in."""

    code = "spc_conflict"
