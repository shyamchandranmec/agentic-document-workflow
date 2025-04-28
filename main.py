#from helper import extract_html_content
import random
#from helper import get_openai_api_key
from workflow import(
    MyWorkflow,
    ProgressEvent,
    TextEvent
)

import asyncio
from llama_index.utils.workflow import draw_all_possible_flows

import load_dotenv

load_dotenv.load_dotenv(override=True)

async def main():
    print("Hello from agentic-document-workflow!")
    basic_workflow = MyWorkflow(timeout=100, verbose=True)
    handler = basic_workflow.run(first_input="Hello")

    async for ev in handler.stream_events():
        if isinstance(ev, ProgressEvent):
            print(ev.progress_output)
        elif isinstance(ev, TextEvent):
            print(ev.delta, end="", flush=True)
    draw_all_possible_flows(MyWorkflow, filename="basic_workflow.html")

if __name__ == "__main__":
    asyncio.run(main())
