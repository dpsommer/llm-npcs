from npcs.memory import sentiment_analysis


def test_sentiment_analysis():
    polarity, _ = sentiment_analysis("That sounds great, I'll have the stout then.")
    assert polarity > 0.5
