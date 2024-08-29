from pydantic import BaseModel, Field



class NodeIndex(BaseModel):
     
class NodeSummaryIndex(NodeIndex):
    """A """



class EntityIndex(BaseIndex):

class RelationIndex(BaseIndex):

class ConceptIndex(BaseIndex):

class TopicIndex(BaseIndex):

class VectorIndex(BaseIndex):

class GraphIndex(BaseIndex):
    entities: Optional[EntityIndex]
    relations: Optional[RelationIndex]