
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
    Event
)
import random
from llama_index.llms.openai import OpenAI


class LoopEvent(Event):
    loop_output:str

class FirstEvent(Event):
    first_output:str

class SecondEvent(Event):
    second_output:str

class TextEvent(Event):
    delta:str

class ProgressEvent(Event):
    progress_output:str

class MyWorkflow(Workflow):
    @step
    async def step_one(self, ctx: Context, ev: StartEvent | LoopEvent) -> FirstEvent |LoopEvent:
        ctx.write_event_to_stream(ProgressEvent(progress_output="Progress: Starting step one..."))
        if random.randint(0, 1) == 0:
            ctx.write_event_to_stream(ProgressEvent(progress_output="Progress: Looping in step one..."))
            return LoopEvent(loop_output="Looping... Back to step one")
        else :
            ctx.write_event_to_stream(ProgressEvent(progress_output="Progress: Completing step one..."))
            return FirstEvent(first_output="First step completed")

    @step
    async def step_two(self, ctx: Context, ev: FirstEvent) -> SecondEvent:
        ctx.write_event_to_stream(ProgressEvent(progress_output="Progress: Starting step two..."))
        llm = OpenAI(model = "gpt-4o-mini")
        generator = await llm.astream_complete("Write a short story about a cat")
        async for response in generator:
            ctx.write_event_to_stream(TextEvent(delta=response.delta))
        return SecondEvent(second_output="Second step completed", response = str(response))

    @step
    async def step_three(self, ctx: Context, ev: SecondEvent) -> StopEvent:
        ctx.write_event_to_stream(ProgressEvent(progress_output="Progress: Starting step three..."))
        print(ev.response)
        ctx.write_event_to_stream(ProgressEvent(progress_output="Progress: Completing step three..."))
        return StopEvent(result = "Workflow completed")
