"""
Sentiment Harness - daily AAPL news collector + FinBERT scorer.

Runs on a schedule (GitHub Actions) or manually. Each run:
  1. Fetches fresh headlines from multiple free RSS feeds (no API keys).
  2. Deduplicates against the accumulated history (only NEW headlines are scored).
  3. Scores new headlines with ProsusAI/finbert -> score = P(pos) - P(neg) in [-1, +1].
  4. Rebuilds the daily aggregate file in the exact thesis format:
        date, mean_score, headline_count, std_score
     so it drops straight into the LSTM pipeline as the sentiment feature source.

Files produced (committed back to the repo by the workflow):
  data/AAPL_headlines_scored.csv   raw corpus: date, headline, score
  data/AAPL_sentiment.csv          daily aggregates consumed by the model
"""

import io
import os
import sys
import feedparser
import pandas as pd

TICKER = os.environ.get("TICKER", "AAPL")

FEEDS = [
    "https://news.google.com/rss/search?q={t}+stock&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q={t}+earnings+OR+{t}+iPhone+OR+{t}+Apple+Inc&hl=en-US&gl=US&ceid=US:en",
    "https://finance.yahoo.com/rss/headline?s={t}",
]

HEADLINES_CSV = os.path.join("data", f"{TICKER}_headlines_scored.csv")
DAILY_CSV = os.path.join("data", f"{TICKER}_sentiment.csv")


def fetch_headlines(ticker: str = TICKER, max_per_feed: int = 100) -> pd.DataFrame:
    rows = []
    for tpl in FEEDS:
        feed = feedparser.parse(tpl.format(t=ticker))
        for e in feed.entries[:max_per_feed]:
            pub = getattr(e, "published_parsed", None)
            if not pub:
                continue
            ts = pd.Timestamp(*pub[:6], tz="UTC").tz_convert("America/New_York").normalize()
            trade_date = ts.tz_localize(None).date()
            title = e.title.rsplit(" - ", 1)[0].strip()
            if title:
                rows.append({"date": trade_date, "headline": title})
    df = pd.DataFrame(rows, columns=["date", "headline"])
    return df.drop_duplicates(subset=["headline"])


def load_history() -> pd.DataFrame:
    if os.path.exists(HEADLINES_CSV):
        old = pd.read_csv(HEADLINES_CSV, parse_dates=["date"])
        return old
    return pd.DataFrame(columns=["date", "headline", "score"])


def load_finbert():
    from transformers import pipeline
    return pipeline("text-classification", model="ProsusAI/finbert", truncation=True, max_length=64)


LABEL_MAP = {"positive": 1, "pos": 1, "negative": -1, "neg": -1, "neutral": 0, "neu": 0}


def score_new(clf, new_df: pd.DataFrame) -> pd.DataFrame:
    texts = new_df["headline"].astype(str).tolist()
    outs = clf(texts, batch_size=32)
    new_df = new_df.copy()
    new_df["score"] = [LABEL_MAP.get(o["label"].lower(), 0) * o["score"] for o in outs]
    return new_df


def rebuild_daily(history: pd.DataFrame) -> pd.DataFrame:
    g = history.dropna(subset=["score"]).copy()
    g["date"] = pd.to_datetime(g["date"]).dt.date
    daily = g.groupby("date")["score"].agg(
        mean_score="mean", headline_count="count", std_score=lambda s: s.std(ddof=0)
    ).reset_index()
    daily["std_score"] = daily["std_score"].fillna(0.0)
    return daily.sort_values("date")


def main() -> int:
    print(f"[harness] collecting for {TICKER}")
    fresh = fetch_headlines()
    history = load_history()
    known = set(history["headline"].astype(str)) if len(history) else set()
    new = fresh[~fresh["headline"].isin(known)].drop_duplicates(subset=["headline"])

    print(f"[harness] fetched {len(fresh)} unique today | {len(new)} genuinely new | history {len(history)}")

    if len(new):
        clf = load_finbert()
        new = score_new(clf, new)
        combined = pd.concat([history, new], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"]).dt.date
        combined = combined.drop_duplicates(subset=["headline"]).sort_values("date")
    else:
        combined = history.copy()

    os.makedirs("data", exist_ok=True)
    combined.to_csv(HEADLINES_CSV, index=False)

    daily = rebuild_daily(combined)
    daily.to_csv(DAILY_CSV, index=False)

    last5 = daily.tail(5)
    print(f"[harness] daily aggregates now cover {daily['date'].min()} -> {daily['date'].max()} ({len(daily)} days)")
    print(last5.to_string(index=False))

    if len(new) == 0:
        print("[harness] nothing new scored this run (normal on quiet days)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
