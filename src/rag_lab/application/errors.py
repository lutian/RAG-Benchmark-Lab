"""Application errors. Inbound adapters translate them into transport-level responses."""


class RagLabError(Exception):
    """Base class for expected, user-facing failures."""


class UnsupportedDocumentError(RagLabError):
    """The file type is not one of the supported formats (PDF, Markdown, TXT)."""


class DocumentParsingError(RagLabError):
    """The file has a supported type but its content cannot be turned into text."""


class UploadTooLargeError(DocumentParsingError):
    """The file exceeds the configured upload limit."""


class NotFoundError(RagLabError):
    """A requested entity does not exist (or is not visible in the given collection)."""


class ConflictError(RagLabError):
    """The operation conflicts with the current state (duplicate name, unindexed chunk set...)."""


class EmbeddingProviderError(RagLabError):
    """The embeddings provider failed or returned vectors that do not match the spec."""


class InvalidRequestError(RagLabError):
    """The request is well-formed but its values cannot be used (e.g. a blank query)."""


class RerankerError(RagLabError):
    """The reranker failed (model not installed, API unreachable, malformed response)."""
