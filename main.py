# https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-1-build-a-basic-chatbot


from typing import Annotated
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

memory = MemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
#tool.invoke("What's a 'node' in LangGraph?")

# from langchain_anthropic import ChatAnthropic

# llm = ChatAnthropic(model="claude-3-haiku-20240307")

from langchain_ollama.chat_models import ChatOllama
llm = ChatOllama(base_url='m1.local:11434', model="llama3.2")
# Modification: tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools)
llm = llm_with_tools # jrw


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile(
    checkpointer=memory,
    # This is new!
    interrupt_before=["tools"],
    # Note: can also interrupt __after__ tools, if desired.
    # interrupt_after=["tools"]
)


config = {"configurable": {"thread_id": "1"}}
user_input = "I'm learning LangGraph. Could you do some research on it for me?"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()

snapshot = graph.get_state(config)
print(snapshot.next)
existing_message = snapshot.values["messages"][-1]
print(existing_message.tool_calls)

events = graph.stream(None, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

exit()

user_input = "Remember my name?"

# The config is the **second positional argument** to stream() or invoke()!
# config = {"configurable": {"thread_id": "2"}} different thread id loses that context.
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()

#snapshot = graph.get_state(config)
#print(snapshot)

exit()

from langchain_core.messages import BaseMessage
while True:
    user_input = input("User: ")
    print("User: "+ user_input)
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            if isinstance(value["messages"][-1], BaseMessage):
                print("Assistant:", value["messages"][-1].content)
