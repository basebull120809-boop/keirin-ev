"""追加の視点：落車・欠場明け、選手同士の対戦成績、ラインを組んだ相手との連携

すべて「そのレースより前」の情報だけで作る（当日の結果は使わない）。
  - 落車・欠場明け : 出走表の「今場所・前場所・前々場所」（history）から
  - 対戦成績・連携 : 取得済みの全着順（finish）から日付順に累計
"""
from functools import lru_cache
import numpy as np
import pandas as pd

H2H_K = 6        # 対戦成績の縮め具合（この回数ぶん五分の対戦を足したのと同じ）
SYNC_K = 4       # 連携の縮め具合
LAYOFF_DAYS = 45 # 前走からこれ以上空いたら「休み明け」

# 対戦成績・連携（p_x_h2h / p_x_sync）は検証で市場を上回る差が見られなかったので特徴量には入れない
# （2025-03〜2026-11 の約32万走。pair_features は調査用に残す）
X_FEATURES = ["p_x_fall_last", "p_x_fall_recent", "p_x_absent_last", "p_x_layoff"]
X_LABEL = dict(p_x_fall_last="落車明け（直前の走）", p_x_fall_recent="最近の落車回数",
               p_x_absent_last="欠場明け", p_x_layoff="休み明け（前走から45日以上）",
               p_x_h2h="対戦成績（同じ相手との先着率）", p_x_sync="ラインの連携（前の選手と一緒に3着内）")


def _rkey(df):
    term = pd.to_numeric(df.get("term"), errors="coerce").fillna(0).astype(int).astype(str)
    return df["name"].astype(str) + "|" + df["pref"].astype(str) + "|" + term


def _race_date(rid: pd.Series):
    """race_id（例 5120261125020001）→ 開催初日 + (日目-1)"""
    s = rid.astype(str)
    base = pd.to_datetime(s.str[2:10], format="%Y%m%d", errors="coerce")
    return base + pd.to_timedelta(pd.to_numeric(s.str[10:12], errors="coerce").fillna(1) - 1, unit="D")


# ---------- 落車・欠場明け ----------
def fall_layoff(df: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"p_x_fall_last": 0.0, "p_x_fall_recent": 0.0,
                        "p_x_absent_last": 0.0, "p_x_layoff": 0.0}, index=df.index)
    if history is None or history.empty:
        return out
    h = history.assign(race_id=history["race_id"].astype(str),
                       date=pd.to_datetime(history["date"], errors="coerce"),
                       t=history["rank_txt"].fillna("").astype(str)).dropna(subset=["date"])
    h = h[~h["t"].str.contains("映像")]
    h = h.sort_values(["race_id", "car", "date"])
    last = h.groupby(["race_id", "car"]).tail(1)
    last = last.set_index(["race_id", "car"])
    falls = h[h["t"].str.contains("落")].groupby(["race_id", "car"]).size()
    key = pd.MultiIndex.from_arrays([df["race_id"].astype(str), df["car"]])
    lt = last["t"].reindex(key)
    ld = last["date"].reindex(key)
    out["p_x_fall_last"] = lt.str.contains("落|故").fillna(False).astype(float).values
    out["p_x_absent_last"] = lt.str.contains("欠").fillna(False).astype(float).values
    out["p_x_fall_recent"] = falls.reindex(key).fillna(0).clip(upper=2).values.astype(float)
    gap = (_race_date(df["race_id"]).values - ld.values) / np.timedelta64(1, "D")
    out["p_x_layoff"] = (pd.Series(gap, index=df.index) >= LAYOFF_DAYS).astype(float)
    return out


# ---------- 対戦成績・連携（全着順の累計） ----------
@lru_cache(maxsize=2)
def _pair_tables(dates_key):
    from . import store
    E, F = [], []
    for d in dates_key:                       # オッズは読まない（メモリ節約）
        e1 = store.read(d, "entries"); f1 = store.read(d, "finish")
        if len(e1) and len(f1):
            E.append(e1[[c for c in ["race_id", "car", "name", "pref", "term", "line_id", "line_pos"] if c in e1]])
            F.append(f1[["race_id", "car", "rank"]])
    if not E:
        return None, None
    e, f = pd.concat(E, ignore_index=True), pd.concat(F, ignore_index=True)
    if e.empty or f.empty:
        return None, None
    e = e.assign(race_id=e["race_id"].astype(str))
    f = f.assign(race_id=f["race_id"].astype(str))
    f = f[pd.to_numeric(f["rank"], errors="coerce").notna()]
    x = f[["race_id", "car", "rank"]].merge(
        e[["race_id", "car", "name", "pref", "term", "line_id", "line_pos"]], on=["race_id", "car"])
    x["r"] = _rkey(x)
    x["date"] = _race_date(x["race_id"])
    x = x.drop_duplicates(["race_id", "car"])
    # 対戦：同じレースの全ペア（a が b に先着したか）
    p = x[["race_id", "date", "r", "rank"]].merge(x[["race_id", "r", "rank"]], on="race_id", suffixes=("", "_b"))
    p = p[p["r"] != p["r_b"]]
    p["w"] = (p["rank"] < p["rank_b"]).astype(float)
    h2h = (p.groupby(["r", "r_b", "date"]).agg(n=("w", "size"), w=("w", "sum")).reset_index()
           .sort_values(["r", "r_b", "date"]))
    h2h["key"] = h2h["r"] + "#" + h2h["r_b"]
    h2h["cn"] = h2h.groupby("key")["n"].cumsum()
    h2h["cw"] = h2h.groupby("key")["w"].cumsum()
    # 連携：同じラインで「前の選手」と自分が一緒に3着内に入ったか
    x["top3"] = (x["rank"] <= 3).astype(float)
    front = x[["race_id", "line_id", "line_pos", "r", "top3"]].rename(
        columns={"r": "r_f", "top3": "top3_f", "line_pos": "pos_f"})
    front["line_pos"] = front["pos_f"] + 1
    s = x.merge(front.drop(columns="pos_f"), on=["race_id", "line_id", "line_pos"])
    s["both"] = s["top3"] * s["top3_f"]
    sync = (s.groupby(["r", "r_f", "date"]).agg(n=("both", "size"), b=("both", "sum")).reset_index()
            .sort_values(["r", "r_f", "date"]))
    sync["key"] = sync["r"] + "#" + sync["r_f"]
    sync["cn"] = sync.groupby("key")["n"].cumsum()
    sync["cb"] = sync.groupby("key")["b"].cumsum()
    base = float(s["both"].mean()) if len(s) else 0.2
    return (h2h[["key", "date", "cn", "cw"]].sort_values("date"),
            (sync[["key", "date", "cn", "cb"]].sort_values("date"), base))


def _asof(q, tab):
    q = q.sort_values("date")
    return pd.merge_asof(q, tab, on="date", by="key", allow_exact_matches=False, direction="backward")


def pair_features(df: pd.DataFrame) -> pd.DataFrame:
    from . import store
    out = pd.DataFrame({"p_x_h2h": 0.0, "p_x_sync": 0.0}, index=df.index)
    h2h, sy = _pair_tables(tuple(store.collected_dates()))
    if h2h is None or df.empty:
        return out
    d = df.assign(_i=df.index, race_id=df["race_id"].astype(str), r=_rkey(df),
                  date=_race_date(df["race_id"]))
    # 対戦成績：出走メンバー全員との過去の先着数を合計し、五分からの上積みを縮めて率にする
    q = d[["_i", "race_id", "r", "date"]].merge(d[["race_id", "r"]], on="race_id", suffixes=("", "_b"))
    q = q[q["r"] != q["r_b"]]
    q["key"] = q["r"] + "#" + q["r_b"]
    m = _asof(q[["_i", "key", "date"]], h2h)
    a = m.fillna({"cn": 0, "cw": 0}).groupby("_i").agg(n=("cn", "sum"), w=("cw", "sum"))
    v = (a["w"] - a["n"] / 2) / (a["n"] + H2H_K)
    out.loc[v.index, "p_x_h2h"] = (v * 4).values          # ±0.25 → ±1
    # 連携：ライン2番手以降の選手だけ。前の選手と組んだ過去の「一緒に3着内」率 − 全体平均
    st, base = sy
    fr = d[["race_id", "line_id", "line_pos", "r"]].rename(columns={"r": "r_f"})
    fr["line_pos"] = fr["line_pos"] + 1
    q2 = d[d["line_pos"] >= 2][["_i", "race_id", "line_id", "line_pos", "r", "date"]].merge(
        fr, on=["race_id", "line_id", "line_pos"])
    if len(q2):
        q2["key"] = q2["r"] + "#" + q2["r_f"]
        m2 = _asof(q2[["_i", "key", "date"]], st).fillna({"cn": 0, "cb": 0})
        v2 = (m2["cb"] - m2["cn"] * base) / (m2["cn"] + SYNC_K)
        out.loc[m2["_i"].values, "p_x_sync"] = (v2 * 5).values
    return out


def add(df: pd.DataFrame, history) -> pd.DataFrame:
    fl = fall_layoff(df, history)
    for c in fl:
        df[c] = fl[c].values
    return df
