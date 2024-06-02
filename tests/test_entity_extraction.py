import pytest


@pytest.fixture
def history():
    return """Although he was very busy with his work, Peter had had enough of it.
He and Nancy decided they needed a holiday.
They travelled to Spain because they loved the country very much."""


def test_entity_extraction(nlp):
    doc = nlp.run("This inn's name is the Silver Fox.")
    assert doc.entities == {'the Silver Fox'}


def test_repeated_entity_names(nlp):
    doc = nlp.run("Bob says hello. Jane says 'Hi, Bob'.")
    assert doc.entities == {'Bob', 'Jane'}


def test_entity_extraction_with_context(nlp, history):
    doc = nlp.run(
        message="Their flight took a while, but the weather was beautiful when they landed.",
        context=history
    )
    assert doc.entities == {'Peter', 'Nancy', 'Spain'}


def test_coref_resolution(nlp, history):
    doc = nlp.run(
        message="He spent most of the time on the beach, while she explored the town.",
        context=history
    )
    assert "Peter" in doc.resolved_text and "Nancy" in doc.resolved_text


# TODO: fastcoref is pretty good, but has some pretty gaping holes in its
# resolution. For example, this text is resolved as follows:
#
# Although Peter was very busy with Peter's work, Peter had had enough of it.
# Peter and Nancy decided He and Nancy needed a holiday.
# He and Nancy travelled to Spain because He and Nancy loved Spain very much.
# He and Nancy had a wonderful trip and returned home refreshed.
#
# Each instance of "they" is resolved in the first pass as "He and Nancy",
# despite it later being correctly resolved as "Peter and Nancy".
#
# It would be nice to use one of the spacy models, but both coreferee and
# the spacy-experimental coref models are pinned to older versions of spacy.
def test_plural_coref_resolution(nlp, history):
    doc = nlp.run(
        message="They had a wonderful trip and returned home refreshed.",
        context=history
    )
    assert "Peter and Nancy" in doc.resolved_text
