def test_sentiment_analysis(nlp):
    doc = nlp.run("That sounds great, I'll have the stout then.")
    assert doc.sentiment_analysis.polarity > 0.5
