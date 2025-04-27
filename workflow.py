
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
    Event
)
import random


class LoopEvent(Event):
    loop_output:str

class FirstEvent(Event):
    first_output:str

class SecondEvent(Event):
    second_output:str

class MyWorkflow(Workflow):
    @step
    async def step_one(self, ctx: Context, ev: StartEvent | LoopEvent) -> FirstEvent |LoopEvent:
        if random.randint(0, 1) == 0:
            return LoopEvent(loop_output="Looping... Back to step one")
        else :
            return FirstEvent(first_output="First step completed")

    @step
    async def step_two(self, ctx: Context, ev: FirstEvent) -> SecondEvent:
        print(ev.first_output)
        return SecondEvent(second_output="Second step completed")

    @step
    async def step_three(self, ctx: Context, ev: SecondEvent) -> StopEvent:
        print(ev.second_output)
        return StopEvent(result = "Workflow completed")
