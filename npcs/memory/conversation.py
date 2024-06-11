from langchain.chains.conversation.base import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from transformers import AutoTokenizer

from npcs.utils.constants import (
    CONVERSATION_SUMMARY_TOKEN_LIMIT,
    LLM_FREQUENCY_PENALTY,
    LLM_TEMP,
)

from .index import IndexedMemory
from .search import NPCMemoryVectorStore

MODEL = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# TODO: improve the base prompt with more information about the NPC
# this should probably pull from a separate store containing details
# about the character
messages = [
    {
        "role": "system",
        "content": """Adopt the personality described in the character section below. Respond with a single message to the user.
Consider the user-provided context and conversation history when writing a response. Ensure that the response is coherent and in character.

Character:

{character_sheet}""",
    },
    {
        "role": "user",
        "content": """Conversation History:
{history}
User: {input}""",
    },
]
# apply the chat template appropriate for the HF model
# see https://huggingface.co/docs/transformers/en/chat_templating
chat_prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_tensors="pt"
)
# append character name to assistant generation prompt
# otherwise the model will prepend this to the response
template = chat_prompt + "{name}:"


class Conversation:
    def __init__(self, name: str, index: NPCMemoryVectorStore) -> None:
        llm = HuggingFaceEndpoint(
            repo_id=MODEL,
            temperature=LLM_TEMP,
            repetition_penalty=LLM_FREQUENCY_PENALTY,
        )
        self._conversation_chain = ConversationChain(
            llm=llm,
            verbose=False,
            prompt=PromptTemplate.from_template(template),
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
