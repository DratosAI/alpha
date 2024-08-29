import os
from dotenv import load_dotenv
from typing import Optional
import httpx

import daft
import ray
from openai import AsyncOpenAI
from unitycatalog import AsyncUnitycatalog, DefaultHttpxClient
import mlflow

load_dotenv()


class Config:
    _instance: Optional["Config"] = None

    def __init__(self):
        self._unity_catalog = None
        self._daft = None
        self._ray = None
        self._openai = None
        self._mlflow = None

    @classmethod
    def get_instance(cls) -> "Config":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def unity_catalog(self) -> AsyncUnitycatalog:
        if self._unity_catalog is None:
            UNITY_CATALOG_URL = os.environ.get("UNITY_CATALOG_URL")
            UNITY_CATALOG_TOKEN = os.environ.get("UNITY_CATALOG_TOKEN")

            self._unity_catalog = AsyncUnitycatalog(
                base_url=UNITY_CATALOG_URL,
                token=UNITY_CATALOG_TOKEN,
                timeout=httpx.Timeout(60.0, read=5.0, write=10.0, connect=2.0),
                http_client=DefaultHttpxClient(
                    proxies=os.environ.get("HTTP_PROXY"),
                    transport=httpx.HTTPTransport(local_address="0.0.0.0"),
                ),
            )
        return self._unity_catalog

    def ray(self) -> ray:
        if self._ray is None:
            RAY_RUNNER_HEAD = os.environ.get("RAY_RUNNER_HEAD")
            ray.init(RAY_RUNNER_HEAD, runtime_env={"pip": ["getdaft"]})
            self._ray = ray
        return self._ray

    def daft(self) -> daft:
        if self._daft is None:
            RAY_RUNNER_HEAD = os.environ.get("RAY_RUNNER_HEAD")
            self._daft = daft.context.set_runner_ray(RAY_RUNNER_HEAD)
        return self._daft

    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
            OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
            self._openai = AsyncOpenAI(
                api_key=OPENAI_API_KEY,
                api_base=OPENAI_API_BASE,
                timeout=httpx.Timeout(60.0, read=5.0, write=10.0, connect=2.0),
            )
        return self._openai

    def mlflow(self) -> mlflow:
        if self._mlflow is None:
            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

            self._mlflow = mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        return self._mlflow


config = Config.get_instance()
