from pydantic import Field
from starlette import Request, Response
from typing import Optional, 

from src.core.tools.base import Tool
from src.core.data.structs.artifacts import Artifact



class DocumentConstructionPipeline(Tool):
    """foo"""
    def __init__(self,
        artifact: Artifact = Field(
            ..., 
            description="Versatile model for managing artifacts in the data system, including metadata and payload"
            ),
        stream: Optional[bool] = Field(
            default=False,
            description="",
        ),
        


        ):

        self.super().__init__()
        self.

    def __call__(self, request: Request) -> Response:
        """"""
        if self.stream:
            return self.__async_call__(request)
        else:
            return self.__sync_call__(request)
        
    def __sync_call__(self, request: Request) -> Response:
        """"""
        df = self.ingest()
        df = df.preprocess()
        df = df.segment(self.segmentation_methods)
        df = df.index(self.indices)
        df = df.post_process
        df.persist()

    async def __async_call__(self, request: Request) -> Response:
        """"""
        pass

    def ingest(self, query:  ): 
        pass

    def preprocess(self):

    def segment(self, segmentation_method: ):

    def index(self, indices: List[]):

    def postprocessing(self):

    def persist(self): 



