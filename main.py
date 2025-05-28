from typing import List, Sequence
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from chains import reflection_chain, generate_chain
import time


def generation_node(state: Sequence[BaseMessage]):
    gen_response = generate_chain.invoke({"messages": state})
    print("gen response: ", gen_response)
    print("state: ", state)
    return gen_response

def reflection_node(messages: Sequence[BaseMessage])->List[BaseMessage]:
    
    res = reflection_chain.invoke({"messages": messages})
    print("ref: ", res)
    print("ref state: ", messages)
    return [HumanMessage(content=res.content)]


def build_graph():

    reflect = "reflect"
    generate = "generate"
    builder = MessageGraph()
    builder.add_node(generate, generation_node)
    builder.add_node(reflect, reflection_node)
    builder.set_entry_point(generate)
    def should_continue(state:Sequence[BaseMessage]):
        if len(state)>6:
            return END
        else:
            return reflect
    builder.add_conditional_edges(generate, should_continue)
    builder.add_edge(reflect, generate)

    graph = builder.compile()
    return graph
# print(graph.get_graph().print_ascii())
if __name__=="__main__":
    # print(os.environ["GROQ_API_KEY"])

    inputs = HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)
    graph = build_graph()
    response = graph.invoke(input=inputs)

