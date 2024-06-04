from pathlib import Path

# the gpt temperature ranges from 0.0 to 1.0 and indicates
# the level of generative randomness - between 0.7 and 0.9 is
# good for fantasy, factual outputs should be at or near 0
LLM_TEMP = 0.8
# the gpt frequency penalty ranges from -2.0 to 2.0 and
# helps to prevent repetition of words and phrases in
# sequential responses
LLM_FREQUENCY_PENALTY = 1.0

CONVERSATION_SUMMARY_TOKEN_LIMIT = 2048

# FIXME: change this to a function, e.g. get_index_dir
# that works x-platform (%APPDATA% on Windows) and allows
# the user to specify a custom path.
HOME_DIRECTORY = Path.home()
ROOT_DIRECTORY = HOME_DIRECTORY / ".config" / "npcs"
