# we need to import these for the pipeline components to work
# NOTE: the fastcoref component MUST be imported before spacy
from fastcoref import spacy_component  # noqa # pylint: disable=unused-import
from spacytextblob.spacytextblob import (  # noqa # pylint: disable=unused-import
    SpacyTextBlob,
)

import spacy  # isort:skip
from spacy.tokens import Doc  # isort:skip


class NLPResult:
    def __init__(self, doc: Doc) -> None:
        self._doc = doc
        self.entities = {str(e) for e in doc.ents}
        self.sentiment_analysis = doc._.blob
        self.resolved_text = doc._.resolved_text


class NLPPipeline:
    def __init__(self) -> None:
        nlp = spacy.load("en_core_web_sm")  # base model, entity extraction
        self._textblob = nlp.add_pipe("spacytextblob")  # sentiment analysis
        self._coref = nlp.add_pipe("fastcoref")  # coreference resolution
        self._nlp = nlp

    def run(self, message: str, context: str = None) -> NLPResult:
        return NLPResult(
            self._nlp(
                f"{context}\n{message}",
                component_cfg={"fastcoref": {"resolve_text": True}},
            )
        )


# Use the Global Object Pattern as it's recommended over a singleton
__default = NLPPipeline()


def run(message: str, context: str = None) -> NLPResult:
    return __default.run(message, context)
