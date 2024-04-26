import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, cast, Sequence, Type,Callable


from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_experimental.pydantic_v1 import root_validator,BaseModel, Field, root_validator
from langchain_core.agents import AgentActionMessageLog, AgentFinish

DEFAULT_SYSTEM_TEMPLATE = """You have access to the following tools:

{tools}

You must always select one and only one of these above tools and respond with only a JSON object matching the following schema:

{{
  "tool": <name of the selected tool>,
  "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
}}

In your response, include one tool and one tool only. For process require a follow up tool use, I will reply you the tool result in the next chat. 
"""  # noqa: E501


DEFAULT_RESPONSE_FUNCTION = {
    "name": "__conversational_response",
    "description": (
        "Respond conversationally if no other tools should be called for a given query."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "Conversational response to the user.",
            },
        },
        "required": ["response"],
    },
}

def parse(output):
    # If no function was invoked, return to user
    if "function_call" not in output.additional_kwargs:
        return AgentFinish(return_values={"output": output.content}, log=output.content)

    # Parse out the function call
    function_call = output.additional_kwargs["function_call"]
    name = function_call["name"]
    inputs = json.loads(function_call["arguments"])

    # If the Response function was invoked, return to the user with the function inputs
    if name == "Response":
        return AgentFinish(return_values=inputs, log=str(function_call))
    # Otherwise, return an agent action
    else:
        return AgentActionMessageLog(
            tool=name, tool_input=inputs, log="", message_log=[output]
        )

class OllamaFunctions(BaseChatModel):
    """Function chat model that uses Ollama API."""

    llm: ChatOllama

    tool_system_prompt_template: str

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        values["llm"] = values.get("llm") or ChatOllama(**values, format="json")
        values["tool_system_prompt_template"] = (
            values.get("tool_system_prompt_template") or DEFAULT_SYSTEM_TEMPLATE
        )
        return values

    @property
    def model(self) -> BaseChatModel:
        """For backwards compatibility."""
        return self.llm

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        functions = kwargs.get("functions", [])
        if "function_call" in kwargs:
            functions = [
                fn for fn in functions if fn["name"] == kwargs["function_call"]["name"]
            ]
            if not functions:
                raise ValueError(
                    'If "function_call" is specified, you must also pass a matching \
function in "functions".'
                )
            del kwargs["function_call"]
        elif not functions:
            functions.append(DEFAULT_RESPONSE_FUNCTION)
        system_message_prompt_template = SystemMessagePromptTemplate.from_template(
            self.tool_system_prompt_template
        )
        system_message = system_message_prompt_template.format(
            tools=json.dumps(functions, indent=2)
        )
        if "functions" in kwargs:
            del kwargs["functions"]
        response_message = self.llm.predict_messages(
            [system_message] + messages, stop=stop, callbacks=run_manager, **kwargs
        )
        chat_generation_content = response_message.content
        if not isinstance(chat_generation_content, str):
            raise ValueError("OllamaFunctions does not support non-string output.")
        try:
            parsed_chat_result = json.loads(chat_generation_content)
        except json.JSONDecodeError:
            raise ValueError(
                f'"{self.llm.model}" did not respond with valid JSON. Please try again.'
            )
        called_tool_name = parsed_chat_result["tool"]
        called_tool_arguments = parsed_chat_result["tool_input"]
        called_tool = next(
            (fn for fn in functions if fn["name"] == called_tool_name), None
        )

        if called_tool is None:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="\n The calculation is finished",
                        )
                    )
                ]
            )

        if called_tool_name == DEFAULT_RESPONSE_FUNCTION["name"]:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=called_tool_arguments["response"],
                        )
                    )
                ]
            )

        response_message_with_functions = AIMessage(
            content="Calling function {} with argument {}".format(called_tool_name, called_tool_arguments),
            additional_kwargs={
                "tool_calls": [{
                    "id": 123,
                    'function':{
                    "name": called_tool_name,
                    "arguments": json.dumps(called_tool_arguments)
                    if called_tool_arguments
                    else "",}
                }],
            },
        )

        return ChatResult(
            generations=[ChatGeneration(message=response_message_with_functions)]
        )

    @property
    def _llm_type(self) -> str:
        return "ollama_functions"

    def bind_tools(self, functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]],
        function_call: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        from langchain.chains.openai_functions.base import convert_to_openai_function
        formatted_functions = [convert_to_openai_function(fn) for fn in functions]
        return super().bind(
            functions=formatted_functions,
            **kwargs,
        )
