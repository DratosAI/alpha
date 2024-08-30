import gzip
import hashlib
import mimetypes
import uuid
from datetime import datetime, date, time
from src.core.data.domain.base import DomainObject
from enum import Enum
from typing import Optional

import ulid
from pydantic import Field, HttpUrl, field_validator
import pyarrow as pa

class SensitivityLevel(int, Enum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3
    SECRET = 4
    # Add more levels as needed


class ArtifactError(Exception):
    """Base exception class for Artifact errors."""


class PayloadError(ArtifactError):
    """Raised when there's an issue with the artifact payload."""


class Artifact(DomainObject):
    """
    Versatile model for managing artifacts in a data system.
    ─────────────────────────────────────────────
    Attributes:
      id               │ ULID identifier
      name             │ Artifact name
      artifact_uri     │ Storage URI
      payload          │ Gzipped content
      extension        │ File extension
      mime_type        │ MIME type
      version          │ Artifact version
      size_bytes       │ Size in bytes
      checksum         │ MD5 checksum
      account_id       │ Account UUID
      user_id          │ User UUID
      user_session_id  │ Session ULID
      region_id        │ Geographic region
      is_ai_generated  │ AI generation flag
      sensitivity_level│ Confidentiality level
      context_event_id │ Context event ULID
      created_at       │ Creation timestamp (Generated from ULID ID)
      updated_at       │ Last update timestamp
      metadata         │ Additional metadata
    ─────────────────────────────────────────────
    Methods:
      compress_payload(payload: bytes) -> bytes
        Compress the given payload using gzip
      decompress_payload(payload: bytes) -> bytes
        Decompress the given gzipped payload
      verify_integrity() -> bool
        Verify artifact integrity via checksum
      from_file(file_path: str, **kwargs)
        Create Artifact instance from file
    ─────────────────────────────────────────────
    Validators:
      set_mime_type()  │ Auto-set MIME type
      set_checksum()   │ Compute MD5 checksum
    ─────────────────────────────────────────────
    Usage:
      session = get_session()                                                                                                                                                                                                               │
      artifact = Artifact.from_file(
        "/path/to/file.pdf",
        user_session = session,
        is_ai_generated = False,
        sensitivity_level = 'CONFIDENTIAL'
      )
      is_valid = artifact.verify()
    ─────────────────────────────────────────────
    Notes:
      • Inherits from CustomModel
      • Payload always stored gzipped
      • Handle PayloadError when using payload
    ─────────────────────────────────────────────
    """

    def __init__(self,
        filename: str = Field(
             ...,
             description="Name of the artifact",
             example="Q2 Financial Report.pdf",
             max_length=255
         ),
        payload: Optional[bytes] = Field(
            None,
            description="The artifact object payload, always gzipped",
            example=b"[gzipped content]"
        ),
        extension: str = Field(
            ...,
            description="File extension of the artifact.",
            example="pdf"
        ),
        mime_type: str = Field(
            ...,
            description="MIME type of the artifact",
            example="application/pdf"
        ),
        version: int = Field(
            default=1,
            description="Version of the artifact",
            example=1
        ),
        size_bytes: float = Field(
            ...,
            description="Size of the artifact in bytes",
            example=1048576
        ),
        checksum: str = Field(
            ...,
            description="MD5 checksum of the artifact payload",
            example="d41d8cd98f00b204e9800998ecf8427e"
        ),
        session_id: ulid.ulid = Field(
            ...,
            description="Unique identifier of the user session during which this artifact was created or last modified",
            example="abcdef12-3456-7890-abcd-ef1234567890"
        ),
        account_id: uuid.UUID = Field(
            ...,
            description="Unique identifier of the account associated with this artifact",
            example="98765432-1234-5678-9abc-defghijklmno"
        ),
        user_id: uuid.UUID = Field(
            ...,
            description="Unique identifier of the user who created or last modified this artifact",
            example="12345678-abcd-efgh-ijkl-mnopqrstuvwx"
        ),
        region_id: str = Field(
            ...,
            description="Identifier of the geographic region associated with this artifact",
            example="us-west-2",
            max_length=50
        ),
        is_ai_generated: bool = Field(
            ...,
            description="Indicates whether the artifact was generated by an AI system",
            example=False
        ),
        sensitivity_level: SensitivityLevel = Field(
            ...,
            description="Indicates the sensitivity or confidentiality level of the artifact",
            example=SensitivityLevel.CONFIDENTIAL
        ),
        context_event_id: str = Field(
            default_factory=lambda: ulid.new().str,
            description="ULID Unique identifier for the corresponding context event that triggered the upload of the artifact.",
            example="01F8MECHZCP3QV1Z4T5BA4WJXN"
        ),
        artifact_uri: HttpUrl = Field(
            ...,
            description="URI where the artifact is stored",
            example="gs://your-bucket/artifacts/account_id/user_id/q2_financial_report.pdf.gz"
        )
    ):
        self.file_name = filename
        self.payload = payload
        self.extension = extension
        self.mime_type = mime_type
        self.version = version
        self.size_bytes = size_bytes
        self.checksum = checksum
        self.session_id = session_id
        self.account_id = account_id
        self.user_id = user_id
        self.region_id = region_id
        self.is_ai_generated = is_ai_generated
        self.sensitivity_level = sensitivity_level
        self.context_event_id = context_event_id
        self.artifact_uri = artifact_uri


    @field_validator('mime_type', mode='before', check_fields=False)
    @classmethod
    def set_mime_type(cls, v, values):
        if v:
            return v
        ext = values.get('extension')
        if ext:
            mime_type, _ = mimetypes.guess_type(f"file.{ext}")
            if mime_type:
                return mime_type
            raise ValueError("Unable to determine MIME type")

    @field_validator('checksum', mode='before', check_fields=False)
    @classmethod
    def set_checksum(cls, v, values):
        if v:
            return v
        payload = values.get('payload')
        if payload:
            return hashlib.md5(payload).hexdigest()
        raise ValueError("Unable to compute checksum: no payload provided")

    def compress_payload(self, payload: bytes) -> bytes:
        try:
            return gzip.compress(payload)
        except Exception as e:
            raise PayloadError(f"Failed to compress payload: {str(e)}")

    def decompress_payload(self, compressed_payload: bytes) -> bytes:
        try:
            return gzip.decompress(compressed_payload)
        except Exception as e:
            raise PayloadError(f"Failed to decompress payload: {str(e)}")

    def verify_integrity(self) -> bool:
        if not self.payload:
            raise PayloadError("No payload to verify")
        return self.checksum == hashlib.md5(self.payload).hexdigest()

    @classmethod
    def from_file(cls,
                  file_path: str,
                  is_ai_generated: Optional[bool] = False,
                  **kwargs):
        with open(file_path, 'rb') as file:
            payload = file.read()

        name = kwargs.get('name', file_path.split('/')[-1])
        extension = name.split('.')[-1]
        size_bytes = len(payload)
        checksum = hashlib.md5(payload).hexdigest()
        mime_type, _ = mimetypes.guess_type(file_path)

        return cls(
            name=name,
            extension=extension,
            size_bytes=size_bytes,
            checksum=checksum,
            mime_type=mime_type,
            payload=gzip.compress(payload),
            **kwargs
        )

    def persist(self):
        # Write Payload to Object Store using Apache Arrow
        with pa.ipc.open_stream(self.payload, self.artifact_uri) as sink:
            sink.write(self.payload)
        # Write Metadata to Metadata Store
        with pa.ipc.open_stream(self.metadata, self.artifact_uri) as sink:
            sink.write(self.metadata)
        # Return Artifact
        return self.artifact_uri

    def __repr__(self):
        return f"<Artifact {self.file_name}.{self.extension} ({self.mime_type})>"

    def __str__(self):
        return
    
    # If using Pydantic v2, replace Config with model_config
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            time: lambda v: v.isoformat(),
            bytes: lambda v: v.decode('utf-8', errors='ignore'),
        },
        "populate_by_name": True,
        "use_enum_values": True,
        "validate_assignment": True,
    }



class ArtifactFactory(DomainObject):
    def create_artifact(self, **kwargs):
        return Artifact(**kwargs)

class ArtifactSelector(DomainObject):
    def select_artifact(self, **kwargs):
        return Artifact(**kwargs)

class ArtifactAccessor(DomainObject):
    def access_artifact(self, **kwargs):
        return Artifact(**kwargs)

class ArtifactMutator(DomainObject):
    def mutate_artifact(self, **kwargs):
        return Artifact(**kwargs)

__all__ = ['Artifact', 'ArtifactFactory', 'ArtifactSelector', 'ArtifactAccessor', 'ArtifactMutator']
