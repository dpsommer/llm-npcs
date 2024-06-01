# we need to import these for the pipeline components to work
# NOTE: the fastcoref component MUST be imported before spacy
from fastcoref import spacy_component  # noqa # pylint: disable=unused-import
from spacytextblob.spacytextblob import SpacyTextBlob  # noqa # pylint: disable=unused-import
from spacy.tokens import Doc
import spacy


class NLPResult:
    def __init__(self, doc: Doc) -> None:
        self._doc = doc
        self.entities = {str(e) for e in doc.ents}
        self.sentiment_analysis = doc._.blob
        self.resolved_text = doc._.resolved_text


class NLPPipeline:
    def __init__(self) -> None:
        nlp = spacy.load('en_core_web_sm')  # base model, entity extraction
        self._textblob = nlp.add_pipe('spacytextblob')  # sentiment analysis
        self._coref = nlp.add_pipe('fastcoref')  # coreference resolution
        self._nlp = nlp

    def run(self, message: str, context: str = None) -> NLPResult:
        return NLPResult(self._nlp(
            f'{context}\n{message}',
            component_cfg={"fastcoref": {'resolve_text': True}}
        ))
