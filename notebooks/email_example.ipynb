{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7534db98",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The token has not been saved to the git credentials helper. Pass `add_to_git_credential=True` in this function directly or `--add-to-git-credential` if using via `huggingface-cli` if you want to set the git credential as well.\n",
      "Token is valid (permission: read).\n",
      "Your token has been saved to /teamspace/studios/this_studio/.cache/huggingface/token\n",
      "Login successful\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import dotenv\n",
    "# Load the .env file from the current directory\n",
    "cwd = os.getcwd()\n",
    "dotenv.load_dotenv(f'{cwd}/.env')\n",
    "from huggingface_hub import login\n",
    "\n",
    "hf_token = os.getenv(\"HF_TOKEN\")\n",
    "login(token=hf_token)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "236fa257",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-09-04 14:36:02,563\tINFO worker.py:1783 -- Started a local Ray instance.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO 09-04 14:36:11 llm_engine.py:184] Initializing an LLM engine (v0.5.5) with config: model='NousResearch/Meta-Llama-3.1-8B-Instruct', speculative_config=None, tokenizer='NousResearch/Meta-Llama-3.1-8B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=20000, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=NousResearch/Meta-Llama-3.1-8B-Instruct, use_v2_block_manager=False, enable_prefix_caching=False)\n",
      "INFO 09-04 14:36:13 model_runner.py:879] Starting to load model NousResearch/Meta-Llama-3.1-8B-Instruct...\n",
      "INFO 09-04 14:36:13 weight_utils.py:236] Using model weights format ['*.safetensors']\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "70db164437784a2b877cf35968c397ff",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO 09-04 14:36:28 model_runner.py:890] Loading model weights took 14.9888 GB\n",
      "INFO 09-04 14:36:34 gpu_executor.py:121] # GPU blocks: 1253, # CPU blocks: 2048\n",
      "INFO 09-04 14:36:37 model_runner.py:1181] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.\n",
      "INFO 09-04 14:36:37 model_runner.py:1185] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.\n",
      "INFO 09-04 14:37:01 model_runner.py:1300] Graph capturing finished in 24 secs.\n"
     ]
    }
   ],
   "source": [
    "# %pip install tantivy\n",
    "# %pip install git+https://github.com/Eventual-Inc/Daft.git\n",
    "# %pip install lancedb\n",
    "# %pip install git+https://github.com/auxon/griptape.git\n",
    "# %pip install cloudpickle\n",
    "# %pip install ray\n",
    "# %pip install pyarrow\n",
    "# %pip install pydantic\n",
    "# %pip install attrs\n",
    "# %pip install vllm -U\n",
    "# %pip install outlines\n",
    "#%pip install hf\n",
    "\n",
    "import ray\n",
    "ray.shutdown()\n",
    "ray.init()\n",
    "\n",
    "from outlines import models, generate\n",
    "from outlines.models import VLLM\n",
    "from outlines.processors.structured import JSONLogitsProcessor\n",
    "from transformers import AutoTokenizer\n",
    "\n",
    "vllm: VLLM = models.vllm(model_name=\"NousResearch/Meta-Llama-3.1-8B-Instruct\", max_model_len=20000)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "2af4f743",
   "metadata": {},
   "outputs": [],
   "source": [
    "import outlines.models as models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8773618b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import sqlite3\n",
    "import cloudpickle\n",
    "import ray\n",
    "\n",
    "def serialize_sqlite_connection(conn):\n",
    "    return ray.data.datasource\n",
    "\n",
    "def deserialize_sqlite_connection(path):\n",
    "    return sqlite3.connect(path)\n",
    "\n",
    "# Register the custom serializer with cloudpickle\n",
    "cloudpickle.register_pickle_by_value(sqlite3)\n",
    "cloudpickle.CloudPickler.dispatch[sqlite3.Connection] = serialize_sqlite_connection\n",
    "\n",
    "# Register the custom serializer with Ray\n",
    "ray.util.register_serializer(\n",
    "    sqlite3.Connection,\n",
    "    serializer=serialize_sqlite_connection,\n",
    "    deserializer=deserialize_sqlite_connection\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "d792207d-f0cc-43d5-a3e5-01b26629e638",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pydantic import BaseModel, Field\n",
    "from typing import List, Optional\n",
    "\n",
    "class EmailModel(BaseModel):\n",
    "    sender: str = Field(..., description=\"Sender's email address\")\n",
    "    subject: str = Field(..., description=\"Subject of the email\")\n",
    "    content: str = Field(..., description=\"Content of the email\")\n",
    "    namespace: Optional[str] = Field(default=None, description=\"Namespace for the email\")\n",
    "    meta: Optional[str] = Field(default=None, description=\"Metadata for the email\")\n",
    "    vector: Optional[List[float]] = Field(default=None, description=\"Vector of content for the email\")\n",
    "\n",
    "class EmailListModel(BaseModel):\n",
    "    emails: List[EmailModel] = Field(..., description=\"List of emails\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "40518308-837a-42ce-9f13-52b59fa6b054",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import uuid\n",
    "import lancedb\n",
    "import pyarrow as pa\n",
    "from typing import List, Dict, Optional\n",
    "from attrs import define, field\n",
    "from pydantic import BaseModel\n",
    "from griptape.mixins import SerializableMixin, FuturesExecutorMixin\n",
    "from lancedb.pydantic import pydantic_to_schema\n",
    "from griptape.drivers.embedding.base_embedding_driver import BaseEmbeddingDriver\n",
    "\n",
    "class EmailEntryModel(BaseModel):\n",
    "    id: str = Field(..., description=\"Unique identifier for the email\")\n",
    "    sender: str = Field(..., description=\"Sender's email address\")\n",
    "    subject: str = Field(..., description=\"Subject of the email\")\n",
    "    content: str = Field(..., description=\"Content of the email\")\n",
    "    namespace: Optional[str] = Field(default=None, description=\"Namespace for the email\")\n",
    "    meta: Optional[str] = Field(default=None, description=\"Metadata for the email\")\n",
    "    vector: Optional[List[float]] = Field(default=None, description=\"Vectors of content for the email\")\n",
    "\n",
    "@define\n",
    "class PydanticPyArrowDaftRayLanceDBDriver(SerializableMixin, FuturesExecutorMixin):\n",
    "    lancedb_path: str = field(kw_only=True, default=\"lancedb_dir\", metadata={\"serializable\": True})\n",
    "    table_name: str = field(kw_only=True, default=\"emails\", metadata={\"serializable\": True})\n",
    "\n",
    "    def __attrs_post_init__(self):\n",
    "        # Initialize LanceDB connection\n",
    "        self.lancedb = lancedb.connect(self.lancedb_path)\n",
    "        \n",
    "        # Check if the table exists and delete it if it does\n",
    "        if self.table_name in self.lancedb.table_names():\n",
    "            self.lancedb.drop_table(self.table_name)\n",
    "            print(f\"Dropped existing table: {self.table_name}\")\n",
    "        \n",
    "        # Create LanceDB table using PyArrow schema\n",
    "        schema = pa.schema([\n",
    "            pa.field('id', pa.string()),\n",
    "            pa.field('sender', pa.string()),\n",
    "            pa.field('subject', pa.string()),\n",
    "            pa.field('content', pa.string()),\n",
    "            pa.field('namespace', pa.string()),\n",
    "            pa.field('meta', pa.string()),\n",
    "            pa.field('vector', pa.list_(pa.float32()))  # Add vector column\n",
    "        ])\n",
    "        table = self.lancedb.create_table(self.table_name, schema=schema)\n",
    "        print(f\"Created table with schema: {table.schema}\")\n",
    "\n",
    "\n",
    "    def upsert_email(self, email: EmailModel, *, email_id: Optional[str] = None, namespace: Optional[str] = None, meta: Optional[Dict] = None) -> str:\n",
    "        if email_id is None:\n",
    "            email_id = self._get_default_id(str(email.dict()))\n",
    "\n",
    "        table = self.lancedb.open_table(self.table_name)\n",
    "        print(f\"Table schema before upsert: {table.schema}\")\n",
    "\n",
    "        # Generate a dummy vector (e.g., a list of floats)\n",
    "        vector = [0.0] * 128  # Example: 128-dimensional zero vector\n",
    "\n",
    "        data = EmailEntryModel(\n",
    "            id=email_id,\n",
    "            sender=str(email.sender),\n",
    "            subject=email.subject,\n",
    "            content=email.content,\n",
    "            namespace=email.namespace,\n",
    "            meta=email.meta,\n",
    "            vector=email.vector \n",
    "        )\n",
    "        print(f\"Data to be inserted: {data.model_dump()}\")\n",
    "\n",
    "        # Ensure all fields are included in the dictionary\n",
    "        data_dict = data.model_dump()\n",
    "        for field in ['namespace', 'meta', 'vector']:\n",
    "            if field not in data_dict:\n",
    "                data_dict[field] = None\n",
    "        data_dict['vector'] = vector  # Add the vector to the data\n",
    "\n",
    "        # Define the schema explicitly\n",
    "        schema = pa.schema([\n",
    "            pa.field('id', pa.string()),\n",
    "            pa.field('sender', pa.string()),\n",
    "            pa.field('subject', pa.string()),\n",
    "            pa.field('content', pa.string()),\n",
    "            pa.field('namespace', pa.string()),\n",
    "            pa.field('meta', pa.string()),\n",
    "            pa.field('vector', pa.list_(pa.float32()))  # Ensure vector column is included\n",
    "        ])\n",
    "\n",
    "        # Convert to PyArrow Table with the defined schema\n",
    "        pyarrow_table = pa.Table.from_pydict({k: [v] for k, v in data_dict.items()}, schema=schema)\n",
    "        print(f\"PyArrow table schema: {pyarrow_table.schema}\")\n",
    "        table.add(pyarrow_table, mode=\"overwrite\")\n",
    "\n",
    "        return email_id\n",
    "\n",
    "    def load_email(self, email_id: str, *, namespace: Optional[str] = None) -> Optional[EmailEntryModel]:\n",
    "        table = self.lancedb.open_table(self.table_name)\n",
    "        query = table.search(f\"id == '{email_id}'\")\n",
    "\n",
    "        if namespace:\n",
    "            query = query.filter(f\"namespace == '{namespace}'\")\n",
    "\n",
    "        result = query.to_pandas().to_dict(orient=\"records\")\n",
    "        if result:\n",
    "            return EmailEntryModel(**result[0])\n",
    "        return None\n",
    "\n",
    "    def load_all_emails(self, *, namespace: Optional[str] = None) -> List[EmailEntryModel]:\n",
    "        table = self.lancedb.open_table(self.table_name)\n",
    "\n",
    "        if namespace:\n",
    "            results = table.search(f\"namespace == '{namespace}'\").to_pandas().to_dict(orient=\"records\")\n",
    "        else:\n",
    "            results = table.to_pandas().to_dict(orient=\"records\")\n",
    "\n",
    "        return [EmailEntryModel(**r) for r in results]\n",
    "\n",
    "    def delete_email(self, email_id: str) -> None:\n",
    "        table = self.lancedb.open_table(self.table_name)\n",
    "        table.delete(f\"id == '{email_id}'\")\n",
    "\n",
    "    def query_by_sender(\n",
    "        self,\n",
    "        sender: str,\n",
    "        *,\n",
    "        count: Optional[int] = None,\n",
    "        namespace: Optional[str] = None,\n",
    "    ) -> List[EmailEntryModel]:\n",
    "        table = self.lancedb.open_table(self.table_name)\n",
    "        query = table.search(f\"sender == '{sender}'\", vector_column_name=\"vector\")\n",
    "\n",
    "        if namespace:\n",
    "            query = query.filter(f\"namespace == '{namespace}'\")\n",
    "\n",
    "        query = query.limit(count or 10)\n",
    "        results = query.to_pandas().to_dict(orient=\"records\")\n",
    "\n",
    "        return [EmailEntryModel(**r) for r in results]\n",
    "\n",
    "    def _get_default_id(self, value: str) -> str:\n",
    "        return str(uuid.uuid5(uuid.NAMESPACE_OID, value))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "26c87b5f-e94d-40df-89ae-63fac93524a2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import daft\n",
    "from griptape.tasks import BaseTask\n",
    "from griptape.artifacts import TextArtifact\n",
    "\n",
    "class EmailProcessingWorkflow(BaseTask):\n",
    "    def __init__(self, lancedb_path: str):\n",
    "        super().__init__()\n",
    "        self.lancedb_path = lancedb_path\n",
    "        daft.set_execution_config(enable_native_executor=True)  # Enable Ray execution\n",
    "        \n",
    "\n",
    "    def run(self, input_data: EmailListModel):\n",
    "        # Convert input data to Daft DataFrame\n",
    "        data = {\n",
    "            \"senders\": [email.sender for email in input_data.emails],\n",
    "            \"subjects\": [email.subject for email in input_data.emails],\n",
    "            \"contents\": [email.content for email in input_data.emails],\n",
    "            \"namespace\": [email.namespace for email in input_data.emails],\n",
    "            \"meta\": [email.meta for email in input_data.emails],\n",
    "            \"vector\": [email.vector for email in input_data.emails],\n",
    "        }\n",
    "        df = daft.from_pydict(data)  # Create a LogicalPlanBuilder\n",
    "\n",
    "        # Process data using Daft (this will be executed on Ray)\n",
    "        df = df.with_column(\"domain\", df[\"senders\"].str.split(\"@\").list.get(-1))\n",
    "        df = df.with_column(\"word_count\", df[\"contents\"].str.split(\" \").list.lengths())\n",
    "\n",
    "        # Write results to LanceDB\n",
    "        df.write_lance(self.lancedb_path)\n",
    "\n",
    "        # Convert Daft DataFrame to Pandas DataFrame for display\n",
    "        result_df = df.to_pandas()\n",
    "\n",
    "        return TextArtifact(f\"Processed data:\\n{result_df}\")\n",
    "\n",
    "    def input(self) -> EmailListModel:\n",
    "        return EmailListModel\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "98914b53-4eb5-4f67-9d48-92405139c281",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dropped existing table: emails\n",
      "Created table with schema: id: string\n",
      "sender: string\n",
      "subject: string\n",
      "content: string\n",
      "namespace: string\n",
      "meta: string\n",
      "vector: list<item: float>\n",
      "  child 0, item: float\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processed prompts: 100%|██████████| 1/1 [00:04<00:00,  4.61s/it, est. speed input: 64.82 toks/s, output: 15.82 toks/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{\n",
      "    \"sender\" : \"alice@example.com\"\n",
      "    , \"subject\" : \"Meeting\"\n",
      "    , \"content\" : \"Let\"\n",
      "    , \"namespace\" : \"personal\"\n",
      "    , \"meta\" : \"location:office\"\n",
      "    , \"vector\" : [ 0.1, 0.2, 0.3 ]\n",
      "}\n",
      "Table schema before upsert: id: string\n",
      "sender: string\n",
      "subject: string\n",
      "content: string\n",
      "namespace: string\n",
      "meta: string\n",
      "vector: list<item: float>\n",
      "  child 0, item: float\n",
      "Data to be inserted: {'id': '7420e06a-0a00-5fb6-9b90-cae4e16b16aa', 'sender': 'alice@example.com', 'subject': 'Meeting', 'content': \"Let's meet at 10 AM.\", 'namespace': 'personal', 'meta': '{}', 'vector': [0.1, 0.2, 0.3]}\n",
      "PyArrow table schema: id: string\n",
      "sender: string\n",
      "subject: string\n",
      "content: string\n",
      "namespace: string\n",
      "meta: string\n",
      "vector: list<item: float>\n",
      "  child 0, item: float\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processed prompts: 100%|██████████| 1/1 [00:05<00:00,  5.24s/it, est. speed input: 56.92 toks/s, output: 15.85 toks/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{\n",
      "    \"sender\" : \"bob@example.org\"\n",
      "  ,\n",
      "  \"subject\" : \"Project Update\"\n",
      "  ,\n",
      "  \"content\" : \"The project is on track.\"\n",
      "  ,\n",
      "  \"namespace\" : \"work\"\n",
      "  ,\n",
      "  \"meta\" : \"status:on track\"\n",
      "  ,\n",
      "  \"vector\" : [ 0.4, 0.5, 0.6 ]\n",
      "}\n",
      "Table schema before upsert: id: string\n",
      "sender: string\n",
      "subject: string\n",
      "content: string\n",
      "namespace: string\n",
      "meta: string\n",
      "vector: list<item: float>\n",
      "  child 0, item: float\n",
      "Data to be inserted: {'id': 'd3757b5e-2c95-59ce-8ac5-1c4ea92f8abf', 'sender': 'bob@example.org', 'subject': 'Project Update', 'content': 'The project is on track.', 'namespace': 'work', 'meta': '{}', 'vector': [0.4, 0.5, 0.6]}\n",
      "PyArrow table schema: id: string\n",
      "sender: string\n",
      "subject: string\n",
      "content: string\n",
      "namespace: string\n",
      "meta: string\n",
      "vector: list<item: float>\n",
      "  child 0, item: float\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processed prompts: 100%|██████████| 1/1 [00:04<00:00,  4.37s/it, est. speed input: 68.15 toks/s, output: 15.78 toks/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{\n",
      "    \"sender\" : \"carol@example.net\" ,\n",
      "    \"subject\" : \"Invoice\" ,\n",
      "    \"content\" : \"Please find the invoice attached.\" ,\n",
      "    \"namespace\" : \"work\" ,\n",
      "    \"meta\" : \"status:pending\" ,\n",
      "    \"vector\" : [ 0.7 ]  \n",
      "}\n",
      "Table schema before upsert: id: string\n",
      "sender: string\n",
      "subject: string\n",
      "content: string\n",
      "namespace: string\n",
      "meta: string\n",
      "vector: list<item: float>\n",
      "  child 0, item: float\n",
      "Data to be inserted: {'id': '6e9f3d77-decd-51fd-a3c7-6cb86ecaaffa', 'sender': 'carol@example.net', 'subject': 'Invoice', 'content': 'Please find the invoice attached.', 'namespace': 'work', 'meta': '{}', 'vector': [0.7, 0.8, 0.9]}\n",
      "PyArrow table schema: id: string\n",
      "sender: string\n",
      "subject: string\n",
      "content: string\n",
      "namespace: string\n",
      "meta: string\n",
      "vector: list<item: float>\n",
      "  child 0, item: float\n",
      "[]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a297103690db4cf3a8f03ebf82102780",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Project-WriteLance [Stage:1]:   0%|          | 0/1 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7ca5c8d586d14189a5d4fafe4bced331",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Project [Stage:2]:   0%|          | 0/1 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed data:\n",
      "             senders        subjects                           contents  \\\n",
      "0  alice@example.com         Meeting               Let's meet at 10 AM.   \n",
      "1    bob@example.org  Project Update           The project is on track.   \n",
      "2  carol@example.net         Invoice  Please find the invoice attached.   \n",
      "\n",
      "  namespace meta           vector       domain  word_count  \n",
      "0  personal   {}  [0.1, 0.2, 0.3]  example.com           5  \n",
      "1      work   {}  [0.4, 0.5, 0.6]  example.org           5  \n",
      "2      work   {}  [0.7, 0.8, 0.9]  example.net           5  \n"
     ]
    }
   ],
   "source": [
    "\n",
    "import outlines\n",
    "import outlines.generate\n",
    "from outlines.generate.api import GenerationParameters, SamplingParameters\n",
    "\n",
    "# Example email data\n",
    "input_data = EmailListModel(emails=[\n",
    "    EmailModel(sender=\"alice@example.com\", subject=\"Meeting\", content=\"Let's meet at 10 AM.\", namespace=\"personal\", meta=\"location:office\", vector=[0.1, 0.2, 0.3]),\n",
    "    EmailModel(sender=\"bob@example.org\", subject=\"Project Update\", content=\"The project is on track.\", namespace=\"work\", meta=\"status:on track\", vector=[0.4, 0.5, 0.6]),\n",
    "    EmailModel(sender=\"carol@example.net\", subject=\"Invoice\", content=\"Please find the invoice attached.\", namespace=\"work\", meta=\"status:pending\", vector=[0.7, 0.8, 0.9])\n",
    "])\n",
    "\n",
    "# Initialize the driver\n",
    "driver = PydanticPyArrowDaftRayLanceDBDriver(lancedb_path=\"./lancedb_dir\")\n",
    "\n",
    "# tokenizer = AutoTokenizer.from_pretrained(\"microsoft/Phi-3.5-mini-instruct\")\n",
    "generation_parameters = GenerationParameters(max_tokens=4096, seed=42, stop_at=\"<|endoftext|>\")\n",
    "logits_processor = JSONLogitsProcessor(\n",
    "    schema=EmailModel.model_json_schema(),\n",
    "    tokenizer=vllm.tokenizer,\n",
    "    whitespace_pattern=\"\\s+\")\n",
    "sampling_parameters = SamplingParameters(temperature=1.0, sampler=\"nucleus\")\n",
    "\n",
    "\n",
    "# Update the email processing loop\n",
    "for email in input_data.emails:\n",
    "    prompt = f\"\"\"Summarize this Email: {email} using the following JSON schema: {EmailModel.model_json_schema()}  Only respond with the JSON object.\\n\\n\"\"\"\n",
    "    \n",
    "    # response = outlines.models.vllm.generate(prompt, EmailModel)\n",
    "    response = vllm.generate(prompts=prompt, generation_parameters=generation_parameters, logits_processor=logits_processor, sampling_parameters=sampling_parameters)\n",
    "    print(response)\n",
    "    try:\n",
    "        result = json.loads(response.lstrip().rstrip())\n",
    "        email.namespace = result.get(\"namespace\")\n",
    "        email.meta = json.dumps(result.get(\"metadata\", {}))\n",
    "    except json.JSONDecodeError:\n",
    "        print(f\"Failed to parse JSON for email: {email.subject}\")\n",
    "        email.namespace = \"unknown\"\n",
    "        email.meta = \"{}\"\n",
    "    \n",
    "    driver.upsert_email(email)\n",
    "\n",
    "# Create the FTS index\n",
    "table = driver.lancedb.open_table(driver.table_name)\n",
    "table.create_fts_index(['sender', 'subject', 'content'])  # Add any other relevant fields\n",
    "\n",
    "# Now you can query emails by a specific sender\n",
    "queried_emails = driver.query_by_sender(\"alice@example.com\")\n",
    "\n",
    "# Print the queried result\n",
    "print(queried_emails)\n",
    "\n",
    "# Run the processing workflow using Ray and Daft\n",
    "workflow = EmailProcessingWorkflow(lancedb_path=\"./lancedb_dir\")\n",
    "result_artifact = workflow.run(input_data=input_data)\n",
    "\n",
    "# Output the processed results\n",
    "print(result_artifact.value)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "355f87bd",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
