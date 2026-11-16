"""誰がレースを動かしたか（H＝先行して最終ホームを先頭で通過した選手）

Kドリームスの結果表には H の記号が無いので、勝敗因（「赤板逃切り」「先行捲られ」など）と決まり手から拾う。
  H の選手 : 決まり手が「逃」、または勝敗因に「先行」「逃」を含む（「先行できず」「逃げられ」などは除く）
             1レースに候補が複数いるときは、決まり手「逃」→ 勝敗因に「先行/逃」の順で1人に決める
  動いた時 : 勝敗因の「赤板」「打鐘」「ホーム」
これで「H の選手・その番手は、市場（3連単オッズ）が見ていた勝率より勝っているか」を調べる。
勝てているなら、レース前に「誰が H を取るか」を当てるモデルを作る価値がある。
"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from . import store

NEG = re.compile(r"先行(?:でき|出来)ず|先行され|逃げられ|逃がし|逃され|に先行|先行許")
POS = re.compile(r"先行|逃")


def _load():
    F, E, O = [], [], []
    for d in store.collected_dates():
        f = store.read(d, "finish")
        if f.empty or "reason" not in f or not f["reason"].notna().any() or (f["reason"] == "-").all():
            continue
        F.append(f); E.append(store.read(d, "entries")); O.append(store.read(d, "odds_trifecta"))
    if not F:
        return None
    f, e, o = (pd.concat(x, ignore_index=True) for x in (F, E, O))
    o = o[o["snapshot"] == "final"].copy()
    o["inv"] = 1 / o["odds"]
    o["q"] = o["inv"] / o.groupby("race_id")["inv"].transform("sum")
    o["car"] = o["combo"].str.split("-").str[0].astype(int)
    mw = o.groupby(["race_id", "car"])["q"].sum().rename("mkt").reset_index()
    df = (e[["race_id", "car", "name", "line_id", "line_pos", "line_size", "tactic"]]
          .merge(f[["race_id", "car", "rank", "kimarite", "sb", "reason"]], on=["race_id", "car"])
          .merge(mw, on=["race_id", "car"]))
    return df


def mark_h(df):
    r = df["reason"].fillna("").astype(str)
    cand = (df["kimarite"].fillna("") == "逃") * 2 + (r.str.contains(POS) & ~r.str.contains(NEG)) * 1
    df = df.assign(_c=cand)
    best = df.groupby("race_id")["_c"].transform("max")
    first = df[(df["_c"] == best) & (best > 0)].sort_values(["race_id", "rank"]).groupby("race_id").head(1)
    df["H"] = False
    df.loc[first.index, "H"] = True
    df["h_when"] = np.select([r.str.contains("赤板"), r.str.contains("打鐘"), r.str.contains("ホーム")],
                             ["赤板", "打鐘", "ホーム"], "")
    return df.drop(columns="_c")


def report(out="reports/hrider"):
    df = _load()
    if df is None:
        return "勝敗因がまだありません（python main.py reasonfill で取得）"
    df = mark_h(df)
    df["win"] = df["rank"] == 1
    df["top3"] = df["rank"] <= 3
    ok = df.groupby("race_id")["H"].sum()
    d = df[df["race_id"].isin(ok[ok == 1].index)].copy()
    hl = d[d["H"]][["race_id", "line_id", "tactic"]].rename(columns={"line_id": "hline", "tactic": "h_tactic"})
    d = d.merge(hl, on="race_id")
    inl = (d["line_id"] == d["hline"]) & (d["line_size"] > 1)
    d["role"] = np.select([d["H"], inl & (d["line_pos"] == 2), inl & (d["line_pos"] >= 3)],
                          ["Hの選手", "その番手", "その3番手"], "その他")
    # 3着内の市場確率（3連単オッズから）
    g = d.groupby("role").agg(n=("win", "size"), 勝率=("win", "mean"), 市場の勝率=("mkt", "mean"),
                              三着内=("top3", "mean")).round(3)
    # 記者の展開予想はHを当てていたか
    pred = d[d["tactic"].fillna("").str.contains("先行|押え先|カマシ")]
    hit = pred.groupby("race_id")["H"].max().mean() if len(pred) else float("nan")
    res = dict(n_races=int(d["race_id"].nunique()), n_all=int(df["race_id"].nunique()),
               roles=g.reset_index().to_dict("records"),
               reporter_hit=round(float(hit), 3),
               when=d[d["H"]]["h_when"].value_counts().to_dict())
    Path(out + ".json").parent.mkdir(parents=True, exist_ok=True)
    Path(out + ".json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    L = ["# 誰がレースを動かしたか（H＝先行した選手）", "",
         f"勝敗因があるレース {res['n_all']} のうち、Hの選手を1人に決められたレース {res['n_races']}", "",
         "| 立場 | 人数 | 勝率 | 市場の勝率 | 3着内率 |", "|---|---|---|---|---|"]
    L += [f"| {k} | {r['n']} | {r['勝率']:.1%} | {r['市場の勝率']:.1%} | {r['三着内']:.1%} |" for k, r in g.iterrows()]
    L += ["", f"記者の展開予想（先行・押え先・カマシ）が実際にHだった率: {res['reporter_hit']:.1%}",
          f"動いた時（勝敗因より）: {res['when']}"]
    Path(out + ".md").write_text("\n".join(L), encoding="utf-8")
    return "\n".join(L)
