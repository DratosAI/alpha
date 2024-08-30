import datetime
from typing import Optional, List
from pydantic import Field
from src.core.data.domain.base import DomainObject
from src.core.data.structs.artifacts.artifact import Artifact

class Project(DomainObject):
    """
    This is a generic Project class for organizing tasks.
    """

    def __init__(
            self,
            name: Optional[str] = Field(default=None, description="Description of the project"),
            description: Optional[str] = None,
            tasks: Optional[List] = None,
            workflows: Optional[List] = None,
            artifacts: Optional[List] = None,
            documents: Optional[List] = None,
            graphs: Optional[List] = None,
            agents: Optional[List] = None,
            # TODO: add more attributes as needed
    ):
        self.name = name
        self.description: Optional[str] = description
        self.tasks: Optional[List] = tasks

        self.artifacts: Optional[List] = Field(
            default_factory=list,
            description="List of artifacts associated with the project")
        self.artifacts = artifacts
        self.documents: Optional[List] = documents
        self.graphs: Optional[List] = graphs
        self.agents: Optional[List] = agents
        self.workflows: Optional[List] = workflows

    def add_artifacts(self, artifact: Artifact):
        self.artifacts.append(artifact)
        self.updated_at = datetime.utcnow()

    def update_artifacts(self, artifacts: List[Artifact]):
        for art in artifacts:
            if art.id in self.artifacts.id:
                self.artifacts[self.artifacts.index(art)] = art
                self.updated_at = datetime.utcnow()

    def remove_artifacts(self, artifact_id: str):
        self.artifacts = [artifact for artifact in self.artifacts if artifact.id != artifact_id]
        self.updated_at = datetime.utcnow()
