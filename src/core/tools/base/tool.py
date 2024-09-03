from typing import Callable, Optional, List, Union
from pydantic import Field
from starlette.requests import Request
from starlette.responses import Response
import ray
import daft
from api.config import config

from src.core.data.domain.base.domain_object import (
    DomainObject,
    DomainObjectFactory,
    DomainObjectSelector,
    DomainObjectAccessor,
)

from enum import Enum

import unittest
import asyncio
from unittest.mock import patch, AsyncMock


class ToolTypes(str, Enum):
    python = "python"
    sql = "sql"


class Tool(DomainObject):
    """A Base Tool Class, used by Agents to perform actions"""
    name: Optional[str] = Field(default="General Pyfunc")
    desc: Optional[str] = Field(default=None)
    type: ToolTypes = Field(default=ToolTypes.python)
    function: Union[str, Callable] = Field(default=None)

    __tablename__ = "tools"

    class Config:
        validate_assignment = True

    # TODO: Implement resource handler
    def __call__(self, *args, **kwargs):
        result = self.function(*args, **kwargs)
        return result

    def __str__(self):
        return f"{self.name} ({self.tool_type}): {self.desc}"


class ToolFactory(DomainObjectFactory):
    @staticmethod
    def create_tool(name: str, description: str, func: Callable) -> Tool:
        return Tool.create_new(name=name, desc=description, function=func)

    @staticmethod
    async def get_tool_by_name(name: str) -> Optional[Tool]:
        df = await ToolAccessor().get_dataframe(Tool.__tablename__, Tool.type)
        filtered_df = df.where(df["name"] == name)
        if filtered_df.count().collect()[0] > 0:
            tool_data = filtered_df.collect().to_pydict()[0]
            return Tool.create_from_data(tool_data)
        return None

    @staticmethod
    async def list_tools() -> List[Tool]:
        df = await ToolAccessor().get_dataframe(Tool.__tablename__, Tool.type)
        tools_data = df.collect().to_pydict()
        return [Tool.create_from_data(data) for data in tools_data]


class ToolSelector(DomainObjectSelector):
    @staticmethod
    def by_name(name: str) -> str:
        return f"{ToolSelector.base_query(Tool.__tablename__)} WHERE name = '{name}' AND _type = '{Tool.type}'"

    @staticmethod
    async def list_tools() -> daft.DataFrame:
        table_location = await ToolAccessor().get_table_location(Tool.__tablename__)
        df = config.daft().from_iceberg(table_location)
        return df.where(df["_type"] == Tool.type)

    @staticmethod
    async def search_tools(keyword: str) -> daft.DataFrame:
        df = await ToolSelector.list_tools()
        return df.where(
            df["name"].str.contains(keyword) | df["desc"].str.contains(keyword)
        )


class ToolAccessor(DomainObjectAccessor):
    @classmethod
    async def get_by_id(cls, id: str) -> daft.DataFrame:
        table_location = await cls.get_table_location(Tool.__tablename__)
        df = config.daft().from_iceberg(table_location)
        return df.where((df["_type"] == Tool.type) & (df["id"] == id))

    @classmethod
    async def get_by_name(cls, name: str) -> daft.DataFrame:
        table_location = await cls.get_table_location(Tool.__tablename__)
        df = config.daft().from_iceberg(table_location)
        return df.where((df["_type"] == Tool.type) & (df["name"] == name))

    @classmethod
    async def list_tools(cls) -> daft.DataFrame:
        table_location = await cls.get_table_location(Tool.__tablename__)
        df = config.daft().from_iceberg(table_location)
        return df.where(df["_type"] == Tool.type)

    @classmethod
    async def search_tools(cls, keyword: str) -> daft.DataFrame:
        table_location = await cls.get_table_location(Tool.__tablename__)
        df = config.daft().from_iceberg(table_location)
        return df.where(df["_type"] == Tool.type).where(
            df["name"].str.contains(keyword) | df["desc"].str.contains(keyword)
        )


class TestToolUsage(unittest.TestCase):
    @patch.object(ToolAccessor, "get_table_location", new_callable=AsyncMock)
    @patch("src.core.tools.base.tool.config.daft")
    async def test_tool_operations(self, mock_daft, mock_get_table_location):
        # Mock the daft DataFrame and its methods
        mock_df = AsyncMock()
        mock_df.where.return_value = mock_df
        mock_df.collect.return_value.to_pydict.return_value = [
            {
                "id": "01H1VXVX7XE1QJ1Z2N3Y4K5P6Q",
                "name": "TestTool",
                "desc": "A test tool",
                "_type": "Tool",
            }
        ]
        mock_daft().from_iceberg.return_value = mock_df
        mock_get_table_location.return_value = "mock_table_location"

        # Test get_by_id
        tool = await ToolAccessor.get_by_id("01H1VXVX7XE1QJ1Z2N3Y4K5P6Q")
        self.assertIsInstance(tool, Tool)
        self.assertEqual(tool.name, "TestTool")

        # Test get_by_name
        tool = await ToolAccessor.get_by_name("TestTool")
        self.assertIsInstance(tool, Tool)
        self.assertEqual(tool.desc, "A test tool")

        # Test list_tools
        tools = await ToolAccessor.list_tools()
        self.assertIsInstance(tools, list)
        self.assertEqual(len(tools), 1)
        self.assertIsInstance(tools[0], Tool)

        # Test search_tools
        tools = await ToolAccessor.search_tools("test")
        self.assertIsInstance(tools, list)
        self.assertEqual(len(tools), 1)
        self.assertIsInstance(tools[0], Tool)


if __name__ == "__main__":
    asyncio.run(unittest.main())
