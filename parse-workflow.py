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
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
import os
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse
from llama_index.core.base.base_query_engine import BaseQueryEngine
from load_dotenv import load_dotenv
from llama_index.utils.workflow import draw_all_possible_flows

load_dotenv(override=True)
nest_asyncio.apply()

class ParseFormEvent(Event):
    application_form:str

class QueryEvent(Event):
    field:str
    query:str

class ResponseEvent(Event):
    response:str

class RAGWorkflow(Workflow):
    storage_dir:str =  "./storage_rag"
    llm: OpenAI
    query_engine: BaseQueryEngine
    embed_model: OpenAIEmbedding = OpenAIEmbedding(model_name="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

    @step
    async def set_up(self, ctx: Context, ev: StartEvent)-> ParseFormEvent:

        if not ev.resume_file:
            raise ValueError("No resume file provided")
        if not ev.application_form:
            raise ValueError("No application form provided")
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
        return ParseFormEvent(application_form=ev.application_form)

    @step
    async def parse_form(self, ctx: Context, ev: ParseFormEvent)-> QueryEvent:
        parser = LlamaParse(
            api_key= os.getenv("LLAMA_PARSE_API_KEY"),
            result_type="markdown",
            content_guideline_instruction= "This is a job application form. Create a list of all the fields that need to be filled in.",
            formatting_instruction= "Return the bulleted list of the fields only"
        )
        result = parser.load_data(ev.application_form)
        text = result[0].text
        raw_json = self.llm.complete(f"""
            This is a parsed form.
            Convert it into a JSON object containing only the list of fields to be filled in, in the form {{ fields:[...] }}.
            <form>${text}</form>
            Return the JSON object only. No Markdown
            """)
        fields = json.loads(raw_json.text)["fields"]
        for field in fields:
            ctx.send_event(QueryEvent(
                field= field,
                query=f"Answer the following question based on the resume: {field}"

            ))
        await ctx.set("total_fields", len(fields))
        return

    @step
    async def query(self, ctx: Context, ev: QueryEvent)-> ResponseEvent:
        response = await self.query_engine.aquery(ev.query)
        return ResponseEvent(field = ev.field, response = response.response)


    @step
    async  def fill_in_application(self, ctx: Context, ev: ResponseEvent) -> StopEvent:
        total_fields = await ctx.get("total_fields")
        responses = ctx.collect_events(ev, [ResponseEvent]*total_fields)
        if responses is None:
            return None
        response_list = "\n".join("Field: "+ r.field + "\n" + r.response for r in responses )
        #print(response_list)
        result = self.llm.complete(f"""
            You are given a list of fields in an application form and responses to
            questions about those fields from a resume. Combine the two into a list of
            fields and succinct, factual answers to fill in those fields. 
            If you can't find the answer just add "No result found"

            <responses>
            {response_list}
            </responses>
        """)
        return StopEvent(result=result)

async def main():

    w = RAGWorkflow(timeout=100, verbose=True)
    result = await w.run(
        resume_file="./resume.pdf",
        application_form="./application_form.pdf"
    )
    draw_all_possible_flows(RAGWorkflow, filename="rag_workflow.html")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
