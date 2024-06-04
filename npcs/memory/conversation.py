from langchain.chains.conversation.base import ConversationChain
from langchain.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_huggingface import HuggingFaceEndpoint

from npcs.utils.constants import (
    CONVERSATION_SUMMARY_TOKEN_LIMIT,
    LLM_FREQUENCY_PENALTY,
    LLM_TEMP,
)

from .index import IndexedMemory
from .nlp import NLPPipeline
from .search import default_index

# TODO: improve the base prompt with more information about the NPC
# this should probably pull from a separate store containing details
# about the character
messages = [
    SystemMessagePromptTemplate.from_template(
        """Reply to the input from the Player below as though you are an NPC.
The NPC will provide the Player with contextual information.
The NPC ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:

{history}
"""
    ),
    HumanMessagePromptTemplate.from_template("Player: {input}"),
    AIMessagePromptTemplate.from_template("NPC: "),
]


class Conversation:
    def __init__(self, name: str, nlp=None, index=None) -> None:
        llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            temperature=LLM_TEMP,
            repetition_penalty=LLM_FREQUENCY_PENALTY,
        )
        prompt = ChatPromptTemplate.from_messages(messages=messages)
        self._conversation_chain = ConversationChain(
            llm=llm,
            verbose=False,
            prompt=prompt,
            memory=IndexedMemory(
                name=name,
                index=index or default_index(),
                nlp=nlp or NLPPipeline(),
                llm=llm,
                human_prefix="Player",
                ai_prefix="NPC",
                max_token_limit=CONVERSATION_SUMMARY_TOKEN_LIMIT,
            ),
        )

    def say(self, message: str) -> str:
        return self._conversation_chain.invoke({"input": message})["response"]
