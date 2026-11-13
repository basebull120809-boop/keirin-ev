"""SQLite データベース（出走表・結果・オッズ・購入履歴）"""
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS races (
    race_id   TEXT PRIMARY KEY,      -- 例: 20261113_35_07 (日付_場コード_R)
    date      TEXT NOT NULL,         -- YYYY-MM-DD
    venue     TEXT NOT NULL,         -- 競輪場名
    race_no   INTEGER,
    grade     TEXT,                  -- GP/G1/G2/G3/F1/F2
    session   TEXT,                  -- モーニング/デイ/ナイター/ミッドナイト
    n_riders  INTEGER,
    bank_len  INTEGER,               -- 333/400/500
    straight  REAL,                  -- みなし直線(m)
    wind      REAL,                  -- 風速(m/s)
    note      TEXT                   -- 手入力メモ（荒れ気配など）
);
CREATE TABLE IF NOT EXISTS entries (
    race_id   TEXT NOT NULL,
    car       INTEGER NOT NULL,      -- 車番
    name      TEXT,
    pref      TEXT,                  -- 府県
    class     TEXT,                  -- S1/S2/A1/A2/A3
    style     TEXT,                  -- 逃/両/追
    score     REAL,                  -- 競走得点
    win_rate  REAL,                  -- 勝率(%)
    ren_rate  REAL,                  -- 連対率(%)
    san_rate  REAL,                  -- 3連対率(%)
    back_cnt  INTEGER,               -- B回数
    age       REAL, term REAL, gear REAL,
    s_cnt     INTEGER,               -- S回数
    nige INTEGER, makuri INTEGER, sashi INTEGER, mark INTEGER,   -- 決まり手の回数
    line_id   INTEGER,               -- 同じラインは同じ番号（単騎は固有番号）
    line_pos  INTEGER,               -- ライン内の位置 1=先頭 2=番手 3=3番手
    line_size INTEGER,               -- ラインの人数
    PRIMARY KEY (race_id, car)
);
CREATE TABLE IF NOT EXISTS results (
    race_id   TEXT PRIMARY KEY,
    first     INTEGER, second INTEGER, third INTEGER,
    tri_payout INTEGER               -- 3連単 払戻(100円あたり)
);
CREATE TABLE IF NOT EXISTS odds_trifecta (
    race_id   TEXT NOT NULL,
    combo     TEXT NOT NULL,         -- "1-5-7"
    odds      REAL NOT NULL,
    snapshot  TEXT NOT NULL DEFAULT 'final',   -- final / T-5 など
    PRIMARY KEY (race_id, combo, snapshot)
);
CREATE TABLE IF NOT EXISTS bets (
    bet_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id   TEXT, combo TEXT, stake INTEGER,
    odds_at_bet REAL, model_p REAL, ev REAL,
    mode      TEXT,                  -- paper / real
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    payout    INTEGER                -- 確定後に記入
);
"""


def connect(path="data/keirin.db"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    from .importer import TABLES               # 古いDBに後から増えた列を足す
    for t, cols in TABLES.items():
        have = {r[1] for r in con.execute(f"PRAGMA table_info({t})")}
        for c in cols:
            if c not in have:
                con.execute(f"ALTER TABLE {t} ADD COLUMN {c}")
    con.commit()
    return con
