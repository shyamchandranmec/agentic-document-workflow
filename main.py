from IPython.display import display, HTML, DisplayHandle
#from helper import extract_html_content
import random
#from helper import get_openai_api_key
from workflow import MyWorkflow
import asyncio
from llama_index.utils.workflow import draw_all_possible_flows

async def main():
    print("Hello from agentic-document-workflow!")
    basic_workflow = MyWorkflow(timeout=10, verbose=True)
    result = await basic_workflow.run(first_input="Hello")
    print(result)
    draw_all_possible_flows(MyWorkflow, filename="basic_workflow.html")

if __name__ == "__main__":
    asyncio.run(main())
