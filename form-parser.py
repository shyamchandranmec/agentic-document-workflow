from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
    Event
)
import nest_asyncio
import json
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


async def main():
    parser = LlamaParse(
        api_key= os.getenv("LLAMA_PARSE_API_KEY"),
        result_type="markdown",
        content_guideline_instruction= "This is a job application form. Create a list of all the fields that need to be filled in.",
        formatting_instruction= "Return the bulleted list of the fields only"
    )
    result = parser.load_data("./application_form.pdf")
    text = result[0].text
    print(text)
    llm = OpenAI("gpt-4o-mini")
    print(llm.complete(result[0].text))

    raw_json = llm.complete(f"""
    This is a parsed form.
    Convert it into a JSON object containing only the list of fields to be filled in, in the form {{ fields:[...] }}.
    <form>${text}</form>
    Return the JSON object only. No Markdown
    """)
    print(raw_json.text)
    fields = json.loads(raw_json.text)["fields"]
    print(fields)

if __name__ == "__main__":
    asyncio.run(main())