"""CSV 取込。収集元が何であっても、この4種のCSVに揃えれば分析側は共通で動く。"""
import pandas as pd

TABLES = {
    "races": ["race_id", "date", "venue", "race_no", "grade", "session",
              "n_riders", "bank_len", "straight", "wind", "note"],
    "entries": ["race_id", "car", "name", "pref", "class", "style", "score",
                "win_rate", "ren_rate", "san_rate", "back_cnt",
                "age", "term", "gear", "s_cnt", "nige", "makuri", "sashi", "mark",
                "line_id", "line_pos", "line_size",
                "tip", "evaluation", "kiai", "tactic", "comment"],   # 記者予想（Kドリームス独自）
    "results": ["race_id", "first", "second", "third", "tri_payout"],
    "odds_trifecta": ["race_id", "combo", "odds", "snapshot"],
}


def import_csv(con, table, path):
    if table not in TABLES:
        raise ValueError(f"table は {list(TABLES)} のいずれか")
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in TABLES[table] if c not in df.columns]
    for c in missing:            # 任意列は空で補う
        df[c] = None
    df = df[TABLES[table]]
    if table == "odds_trifecta":
        df["snapshot"] = df["snapshot"].fillna("final")
    cols = ",".join(df.columns)
    ph = ",".join("?" * len(df.columns))
    con.executemany(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({ph})",
                    df.astype(object).where(df.notna(), None).values.tolist())
    con.commit()
    return len(df)


def write_templates(folder):
    for t, cols in TABLES.items():
        pd.DataFrame(columns=cols).to_csv(f"{folder}/{t}.csv", index=False,
                                          encoding="utf-8-sig")
