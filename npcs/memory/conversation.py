from langchain.chains.conversation.base import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from npcs.utils.constants import (
    CONVERSATION_SUMMARY_TOKEN_LIMIT,
    LLM_FREQUENCY_PENALTY,
    LLM_TEMP,
)

from .index import IndexedMemory
from .search import NPCMemoryVectorStore

# TODO: improve the base prompt with more information about the NPC
# this should probably pull from a separate store containing details
# about the character
TEMPLATE = """
<|system|>
Adopt the personality described in the character section below. Respond with a single message to the user.
Consider the user-provided context and conversation history when writing a response. Ensure that the response is coherent and in character.

Character:

Name: {name}
<|user|>
Conversation History:
{history}
User: {input}
<|model|>{name}:
"""


class Conversation:
    def __init__(self, name: str, index: NPCMemoryVectorStore) -> None:
        llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            temperature=LLM_TEMP,
            repetition_penalty=LLM_FREQUENCY_PENALTY,
        )
        prompt = PromptTemplate(
            input_variables=["name", "input", "history"], template=TEMPLATE
        )
        self._name = name
        self._conversation_chain = ConversationChain(
            llm=llm,
            verbose=False,
            prompt=prompt,
            memory=IndexedMemory(
                name=name,
                index=index,
                llm=llm,
                human_prefix="Player",
                ai_prefix="NPC",
                max_token_limit=CONVERSATION_SUMMARY_TOKEN_LIMIT,
            ),
        )

    def say(self, message: str) -> str:
        return self._conversation_chain.invoke({"input": message})["response"]
