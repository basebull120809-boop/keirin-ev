"""日別のCSV(gzip)で保存する。GitHub上に積み上げても軽く、差分も追いやすい。

data/daily/YYYY/MM/YYYY-MM-DD/{races,entries,results,odds_trifecta}.csv.gz
"""
from pathlib import Path
import pandas as pd

ROOT = Path("data/daily")
TABLES = ["races", "entries", "results", "odds_trifecta"]
EXTRA = ["finish", "history"]          # 全着順・上がり / 前場所までの成績


def day_dir(date):
    y, m, _ = date.split("-")
    return ROOT / y / m / date


def has_day(date):
    return (day_dir(date) / "finish.csv.gz").exists()


def save_day(date, frames):
    d = day_dir(date)
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for t in TABLES + EXTRA:
        df = frames.get(t)
        if df is None:
            df = pd.DataFrame()
        df.to_csv(d / f"{t}.csv.gz", index=False, compression="gzip")
        n += len(df)
    return n


def read(date, table):
    try:
        return pd.read_csv(day_dir(date) / f"{table}.csv.gz", dtype={"race_id": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def collected_dates():
    return sorted(p.name for p in ROOT.glob("*/*/*") if (p / "races.csv.gz").exists())


def load_into_db(con):
    """全日分を SQLite に読み込む（分析の前に毎回実行）"""
    from .collector import store
    con.executescript("DELETE FROM races; DELETE FROM entries; "
                      "DELETE FROM results; DELETE FROM odds_trifecta;")
    days = collected_dates()
    for date in days:
        frames = {}
        for t in TABLES:
            try:
                frames[t] = pd.read_csv(day_dir(date) / f"{t}.csv.gz")
            except (FileNotFoundError, pd.errors.EmptyDataError):
                frames[t] = pd.DataFrame()
        store(con, frames)
    return len(days)
