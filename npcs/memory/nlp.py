from typing import Tuple, Set

import spacy

__nlp = None


def default_pipeline() -> spacy.Language:
    global __nlp

    if __nlp:
        return __nlp

    __nlp = spacy.load('en_core_web_sm')  # base model, entity extraction
    __nlp.add_pipe('spacytextblob')  # sentiment analysis
    __nlp.add_pipe('sentencizer')
    __nlp.add_pipe('coreferee')  # coreference identification
    return __nlp


def entity_extraction(message: str, pipeline: spacy.Language = None, context: str = None) -> Set[str]:
    """
    Uses spacy to extract entities from a given message as a comma-separated string.
    """
    if not pipeline:
        pipeline = default_pipeline()
    if context:
        return _coreference_extraction(pipeline=pipeline, message=message, context=context)
    doc = pipeline(f'{context}\n{message}')
    return {str(e) for e in doc.ents}


def _coreference_extraction(pipeline: spacy.Language, message: str, context: str) -> any:
    refs = set()
    tokenizer = pipeline.tokenizer
    tokens_in_context = len(tokenizer(context))
    doc = pipeline(f'{context}\n{message}')
    for chain in doc._.coref_chains:
        # last index of the ref chain is in the message
        if chain[-1].root_index > tokens_in_context:
            tokens = {token.text for token in doc._.coref_chains.resolve(doc[chain[-1].root_index])}
            refs.update(tokens)
            # TODO: replace unspecific refs (he/him etc.)
    return refs


def sentiment_analysis(message: str, pipeline: spacy.Language = None) -> Tuple[float, float]:
    if not pipeline:
        pipeline = default_pipeline()
    doc = pipeline(message)
    return (doc._.blob.polarity, doc._.blob.subjectivity)
