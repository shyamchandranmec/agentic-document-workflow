from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
    Event
)
import nest_asyncio

import asyncio
from llama_index.embeddings.openai import  OpenAIEmbedding
import random
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
import os
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse
from llama_index.core.base.base_query_engine import BaseQueryEngine
from load_dotenv import load_dotenv

load_dotenv(override=True)
nest_asyncio.apply()
class QueryEvent(Event):
    query:str

class RAGWorkflow(Workflow):
    storage_dir:str =  "./storage_rag"
    llm: OpenAI
    query_engine: BaseQueryEngine
    embed_model: OpenAIEmbedding = OpenAIEmbedding(model_name="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

    @step
    async def set_up(self, ctx: Context, ev: StartEvent)-> QueryEvent:

        if not ev.resume_file:
            raise ValueError("No resume file provided")
        self.llm = OpenAI("gpt-4o-mini")
        if os.path.exists(self.storage_dir):
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            print("Loading index from storage")
            index = load_index_from_storage(storage_context=storage_context)
        else:
            documents = LlamaParse(
                api_key= os.getenv("LLAMA_PARSE_API_KEY"),
                result_type="markdown",
                content_guideline_instruction= "This is a resume. Gather related facts together and format it as a json"
            ).load_data(ev.resume_file)
            print("Fresh Embedding and storing it in storage")
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=self.embed_model
            )
            index.storage_context.persist(persist_dir=self.storage_dir)

        self.query_engine = index.as_query_engine(llm=self.llm, similarity_top_k=5)
        return QueryEvent(query=ev.query)

    @step
    async def query(self, ctx: Context, ev: QueryEvent)-> StopEvent:
        response = await self.query_engine.aquery(ev.query)
        return StopEvent(result = response.response)

async def main():
    w = RAGWorkflow(timeout=100, verbose=True)
    result = await w.run(
        resume_file="./resume.pdf",
        query="How many years of experience does the applicant have?"
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
