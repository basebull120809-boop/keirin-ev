"""自動収集アダプタ。

収集元サイトごとにクラスを1つ作り、fetch_day() が4種の DataFrame を返せば
あとは DB へ自動登録される。サイトのHTML構造は変わるので、
最初の1回は実際のページを見てセレクタを合わせる必要がある。

注意:
- 各サイトの利用規約・robots.txt を必ず確認し、許可された範囲で使う
- アクセス間隔は最低でも数秒空ける（SLEEP_SEC）
- オッズは締切直前に変動するので、検証には「最終オッズ」を使う
"""
import time
import pandas as pd

SLEEP_SEC = 3.0


class BaseCollector:
    name = "base"

    def fetch_day(self, date: str) -> dict:
        """date: 'YYYY-MM-DD'。戻り値: {'races': df, 'entries': df,
        'results': df, 'odds_trifecta': df}"""
        raise NotImplementedError

    def get(self, session, url):
        time.sleep(SLEEP_SEC)
        r = session.get(url, timeout=30,
                        headers={"User-Agent": "keirin-ev research (personal use)"})
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text


class CsvFolderCollector(BaseCollector):
    """フォルダに置いた CSV を取り込むだけの収集元（手動ダウンロード運用・テスト用）。
    folder/YYYY-MM-DD/{races,entries,results,odds_trifecta}.csv を読む。"""
    name = "csvfolder"

    def __init__(self, folder):
        self.folder = folder

    def fetch_day(self, date):
        out = {}
        for t in ["races", "entries", "results", "odds_trifecta"]:
            p = f"{self.folder}/{date}/{t}.csv"
            try:
                out[t] = pd.read_csv(p, encoding="utf-8-sig")
            except FileNotFoundError:
                out[t] = pd.DataFrame()
        return out


class WebCollectorTemplate(BaseCollector):
    """Web収集のひな型。URL とパース処理を収集元に合わせて実装する。
    requests + BeautifulSoup を想定（pip install requests beautifulsoup4 lxml）。"""
    name = "web"
    RACE_LIST_URL = "https://example.com/keirin/{date}"   # ←収集元に合わせて変更

    def fetch_day(self, date):
        import requests
        from bs4 import BeautifulSoup  # noqa: F401
        s = requests.Session()
        html = self.get(s, self.RACE_LIST_URL.format(date=date.replace("-", "")))
        raise NotImplementedError(
            "収集元ページのHTMLに合わせて races/entries/results/odds を抽出してください。"
            f"取得したHTML長: {len(html)}")


def store(con, frames: dict):
    from .importer import TABLES
    n = 0
    for t, df in frames.items():
        if df is None or df.empty:
            continue
        for c in TABLES[t]:
            if c not in df.columns:
                df[c] = None
        df = df[TABLES[t]]
        cols = ",".join(df.columns)
        ph = ",".join("?" * len(df.columns))
        con.executemany(f"INSERT OR REPLACE INTO {t} ({cols}) VALUES ({ph})",
                        df.astype(object).where(df.notna(), None).values.tolist())
        n += len(df)
    con.commit()
    return n
