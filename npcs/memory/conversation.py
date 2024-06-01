from langchain.chains.conversation.base import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint

from .index import IndexedMemory
from .search import default_index
from npcs.utils.constants import (
    LLM_TEMP,
    LLM_FREQUENCY_PENALTY,
    CONVERSATION_SUMMARY_TOKEN_LIMIT,
)

TEMPLATE = """The following is a conversation between a Player and an NPC.
The NPC is wary of the player but not hostile, and will provide the Player with contextual information.
The NPC ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:

{history}

Conversation:
Player: {input}
NPC:"""


class Conversation:
    def __init__(self, name: str, index=None) -> None:
        llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            temperature=LLM_TEMP,
            repetition_penalty=LLM_FREQUENCY_PENALTY,
        )
        prompt = PromptTemplate(template=TEMPLATE, input_variables=["history", "input"])
        self._conversation_chain = ConversationChain(
            llm=llm,
            verbose=False,
            prompt=prompt,
            memory=IndexedMemory(
                name=name,
                index=index or default_index(),
                llm=llm,
                human_prefix='Player',
                ai_prefix='NPC',
                max_token_limit=CONVERSATION_SUMMARY_TOKEN_LIMIT,
            ),
        )

    def say(self, message: str) -> str:
        return self._conversation_chain.invoke({"input": message})
