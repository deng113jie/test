from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate

# model = ChatOllama(model='llama3')
model = OllamaFunctions(model='mistral', temperature=0)

prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant, return in JSON format."),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent


tools = [multiply, add, exponentiate]
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "Multiply 3 by 18, add the result to 30, and finally exponentiate that result to 8." +
                                "Remember return in JSON format. In your response, include one tool and one tool only. For process require a follow up tool use, I will reply you the tool result in the next chat."})
