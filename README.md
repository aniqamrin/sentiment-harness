# Sentiment Harness — automated news collector for the FYP pipeline

Replaces the thesis's manual headline pasting with a scheduled agent that builds a
proper historical sentiment corpus day by day. This is the fix for the empty-corpus
weakness documented in Chapter 4 of the report.

## How it works

```
RSS feeds (Google News x2 + Yahoo Finance)     <- free, no API keys
        |
        v
dedupe vs accumulated corpus (only NEW headlines scored)
        |
        v
ProsusAI/finBERT -> score = P(pos) - P(neg) in [-1, +1]
        |
        v
data/AAPL_headlines_scored.csv                 <- full raw corpus (grows daily)
data/AAPL_sentiment.csv                        <- daily aggregates, THESIS FORMAT:
                                                  date, mean_score, headline_count, std_score
```

The GitHub Action runs at 21:00 UTC every day (after US close) and auto-commits the
updated CSVs back to this repo. You never touch anything.

## One-time setup (~5 minutes)

1. Create a new **public** GitHub repo (e.g. `sentiment-harness`).
2. Upload everything in this folder to it — make sure `.github/workflows/collect-news.yml` is included.
3. Repo -> Settings -> Actions -> General -> Workflow permissions ->
   select **Read and write permissions** -> Save.
4. Go to the Actions tab -> "Daily sentiment collection" -> **Run workflow** to test it now.
5. Done. Every day the corpus grows automatically.

## Using the output in the model

Point your training pipeline at `data/AAPL_sentiment.csv` (same schema as
`LTSM_STOCK/sentiment_scores/sentiment_scores/AAPL_sentiment.csv`, so it is a drop-in
replacement), or download the CSV from the repo whenever you retrain.

In the Colab notebook you can load it directly from GitHub raw:

```python
url = "https://raw.githubusercontent.com/<user>/sentiment-harness/main/data/AAPL_sentiment.csv"
scored_history = pd.read_csv(url)
```

## Other tickers

Add another job or change the `TICKER` env var in the workflow (MSFT, TSLA, NVDA,
GOOGL...). Each ticker gets its own pair of CSVs.
