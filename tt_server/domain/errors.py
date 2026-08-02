class InferenceError(Exception):
    """Base error exposed by the application layer."""


class InvalidRequest(InferenceError):
    pass


class UnknownMode(InferenceError):
    pass


class CapabilityUnavailable(InferenceError):
    pass


class RuntimeSwitching(InferenceError):
    pass


class RuntimeNotReady(InferenceError):
    pass
