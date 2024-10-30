import asyncio

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage, AnyMessage

"""
This script build an agent by langgraph and stream LLM tokens
pip install langchain==0.2.16
pip install langgraph==0.2.34
pip install langchain_openai==0.1.9
"""


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def search(query: str):
    """Call to surf the web."""
    return ["Cloudy with a chance of hail."]


tools = [search]

model = ChatOpenAI(
    temperature=0,
    # model="glm-4",
    model="GLM-4-Flash",
    openai_api_key="[You Key]",
    # openai_api_base="https://open.bigmodel.cn/api/paas/v4/", #使用智谱官方提供的是正常流式输出
    openai_api_base="You url by glm_server.py ",
    streaming=True
)


class Agent:

    def __init__(self, model, tools, system=""):
        self.system = system
        workflow = StateGraph(State)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", ToolNode(tools))
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            self.should_continue,
            # Next we pass in the path map - all the nodes this edge could go to
            ["tools", END],
        )
        workflow.add_edge("tools", "agent")
        self.model = model.bind_tools(tools)
        self.app = workflow.compile()

    def should_continue(self, state: State):
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return END
        # Otherwise if there is, we continue
        else:
            return "tools"

    async def call_model(self, state: State, config: RunnableConfig):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        response = await self.model.ainvoke(messages, config)
        # We return a list, because this will get added to the existing list
        return {"messages": response}

    async def query(self, user_input: str):
        inputs = [HumanMessage(content=user_input)]
        first = True
        async for msg, metadata in self.app.astream({"messages": inputs}, stream_mode="messages"):
            if msg.content and not isinstance(msg, HumanMessage):
                # 这里可以看出是否正常流式输出
                print(msg.content, end="|", flush=True)

            if isinstance(msg, AIMessageChunk):
                if first:
                    gathered = msg
                    first = False
                else:
                    gathered = gathered + msg

                if msg.tool_call_chunks:
                    print('tool_call_chunks...', gathered.tool_calls)


if __name__ == '__main__':

    input = "what is the weather in sf"
    prompt = """
    You are smart research assistant. Use the search engine ...
    """
    agent = Agent(model, tools, prompt)
    asyncio.run(agent.query(input))

