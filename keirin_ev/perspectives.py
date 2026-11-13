"""予想の「視点」を選手ごとの数値にする（1着モデルの特徴量になる）

  S（スタート）   : 直近4か月で誘導員の後ろ（スタート）を取った回数
  B（バック）     : バック線を先頭で通過した回数 = 先行・まくりで主導権を取る力
  前の先行力     : 番手・3番手から見た、ライン先頭のB回数（前が駆けるほど番手は有利）
  自力型×先頭     : 逃げ・まくりの決まり手の割合 × ライン先頭
  差し型×番手     : 差し・マークの決まり手の割合 × 番手
  番手巧者       : 過去に番手を回ったとき、市場の見積もりより何回多く3着内に来たか
                   （その日より前のレースだけで計算。回数が少ないと0に寄せる）
  上がり         : 前場所までの「最後の200m」の平均タイム（速いほど＋）
  調子           : 前場所までの直近の着順の平均（良いほど＋）
  地元           : 選手の府県と競輪場の府県が同じ

※ H（ホーム線の先頭通過回数）はKドリームスの出走表に無いため使っていない。
"""
from functools import lru_cache
import numpy as np
import pandas as pd

VENUE_PREF = dict(
    hakodate="北海道", aomori="青森", iwakitaira="福島", yahiko="新潟", maebashi="群馬",
    toride="茨城", utsunomiya="栃木", omiya="埼玉", seibuen="埼玉", keiokaku="東京",
    tachikawa="東京", matsudo="千葉", chiba="千葉", kawasaki="神奈川", hiratsuka="神奈川",
    odawara="神奈川", ito="静岡", shizuoka="静岡", nagoya="愛知", gifu="岐阜",
    ogaki="岐阜", toyohashi="愛知", toyama="富山", matsusaka="三重", yokkaichi="三重",
    fukui="福井", nara="奈良", mukomachi="京都", wakayama="和歌山", kishiwada="大阪",
    tamano="岡山", hiroshima="広島", hofu="山口", takamatsu="香川", komatsushima="徳島",
    kochi="高知", matsuyama="愛媛", kokura="福岡", kurume="福岡", takeo="佐賀",
    sasebo="長崎", beppu="大分", kumamoto="熊本")

BANATE_K = 8          # 番手巧者の縮め具合（この回数ぶん0の実績を足したのと同じ）
NUM_COLS = ["s_cnt", "back_cnt", "nige", "makuri", "sashi", "mark"]


def rider_key(df):
    term = pd.to_numeric(df.get("term"), errors="coerce").fillna(0).astype(int).astype(str)
    return df["name"].astype(str) + "|" + df["pref"].astype(str) + "|" + term


# ---------- 番手巧者（過去の実績） ----------
@lru_cache(maxsize=4)
def _banate_cum(dates_key):
    """選手ごと・日付ごとの「番手での 3着内 − 市場の見積もり」の累計"""
    from .insights import rider_table
    x = rider_table()
    if x is None or x.empty:
        return pd.DataFrame(columns=["rider", "date", "cn", "cd"])
    b = x[x["位置"] == "番手"].copy()
    b["rider"] = rider_key(b)
    b["diff"] = b["top3"].astype(float) - b["mkt_top3"].fillna(0)
    g = (b.groupby(["rider", "date"]).agg(n=("car", "size"), d=("diff", "sum"))
         .reset_index().sort_values(["rider", "date"]))
    g["cn"] = g.groupby("rider")["n"].cumsum()
    g["cd"] = g.groupby("rider")["d"].cumsum()
    g["date"] = pd.to_datetime(g["date"])
    return g[["rider", "date", "cn", "cd"]]


def banate_skill(keys: pd.DataFrame) -> pd.Series:
    """keys: rider, date → その日より前の番手実績から求めた1走あたりの上積み"""
    from . import store
    cum = _banate_cum(tuple(store.collected_dates()))
    out = pd.Series(0.0, index=keys.index)
    if cum.empty or keys.empty:
        return out
    k = keys.assign(_i=keys.index, date=pd.to_datetime(keys["date"])).sort_values("date")
    m = pd.merge_asof(k, cum.sort_values("date"), on="date", by="rider",
                      allow_exact_matches=False, direction="backward")
    v = (m["cd"] / (m["cn"] + BANATE_K)).fillna(0).values
    out.loc[m["_i"].values] = v
    return out


# ---------- 前場所までの成績 ----------
def history_stats(history: pd.DataFrame, last_n=6) -> pd.DataFrame:
    cols = ["race_id", "car", "agari_avg", "form_avg"]
    if history is None or history.empty:
        return pd.DataFrame(columns=cols)
    h = history.copy()
    h["race_id"] = h["race_id"].astype(str)
    h = h.sort_values("date")
    ag = h.dropna(subset=["agari"])
    ag = ag[(ag["agari"] > 8) & (ag["agari"] < 16)]
    a = ag.groupby(["race_id", "car"]).tail(last_n).groupby(["race_id", "car"])["agari"].mean()
    rk = h.dropna(subset=["rank"])
    rk = rk[rk["rank"] <= 9]
    f = rk.groupby(["race_id", "car"]).tail(last_n).groupby(["race_id", "car"])["rank"].mean()
    return pd.DataFrame({"agari_avg": a, "form_avg": f}).reset_index()


def load_history(race_ids, dates):
    from . import store
    H = [store.read(d, "history") for d in sorted(set(dates))]
    H = [h for h in H if len(h)]
    if not H:
        return pd.DataFrame()
    h = pd.concat(H, ignore_index=True)
    h["race_id"] = h["race_id"].astype(str)
    return h[h["race_id"].isin(set(map(str, race_ids)))]


# ---------- まとめて付ける ----------
def _z(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0


def add(entries: pd.DataFrame, races: pd.DataFrame, history=None) -> pd.DataFrame:
    """出走表に視点の列（p_*）を足して返す"""
    df = entries.copy()
    df["race_id"] = df["race_id"].astype(str)
    r = races[["race_id", "date", "venue"]].copy()
    r["race_id"] = r["race_id"].astype(str)
    df = df.drop(columns=[c for c in ("date", "venue") if c in df]).merge(r, on="race_id", how="left")
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    g = df.groupby("race_id")
    df["p_s"] = g["s_cnt"].transform(_z).fillna(0)
    df["p_b"] = g["back_cnt"].transform(_z).fillna(0)
    # 前の先行力：同じラインの先頭（line_pos=1）のB
    head = df[df["line_pos"] == 1][["race_id", "line_id", "p_b"]].rename(columns={"p_b": "_hb"})
    df = df.merge(head.drop_duplicates(["race_id", "line_id"]), on=["race_id", "line_id"], how="left")
    follow = (df["line_pos"] >= 2) & (df["line_size"].fillna(1) > 1)
    df["p_head_b"] = np.where(follow, df["_hb"].fillna(0), 0.0)
    tot = df[["nige", "makuri", "sashi", "mark"]].fillna(0).sum(1) + 1
    is_head = (df["line_pos"] == 1) & (df["line_size"].fillna(1) > 1)
    df["p_jiriki_head"] = np.where(is_head, (df["nige"].fillna(0) + df["makuri"].fillna(0)) / tot, 0.0)
    df["p_oikomi_2nd"] = np.where(df["line_pos"] == 2, (df["sashi"].fillna(0) + df["mark"].fillna(0)) / tot, 0.0)
    # 番手巧者（その日より前の実績）
    if "name" in df and "pref" in df:
        keys = pd.DataFrame({"rider": rider_key(df), "date": df["date"]}, index=df.index)
        df["banate_plus"] = banate_skill(keys.dropna(subset=["date"]))
        df["banate_plus"] = df["banate_plus"].fillna(0)
    else:
        df["banate_plus"] = 0.0
    df["p_banate"] = np.where(df["line_pos"] == 2, df["banate_plus"] * 10, 0.0)   # 0.1 = 1走で+0.1回 → 1
    # 上がり・調子
    if history is None:
        history = load_history(df["race_id"].unique(), df["date"].dropna())
    hs = history_stats(history)
    hs["race_id"] = hs["race_id"].astype(str)
    df = df.merge(hs, on=["race_id", "car"], how="left")
    g = df.groupby("race_id")
    df["p_agari"] = -g["agari_avg"].transform(_z).fillna(0)
    df["p_form"] = -g["form_avg"].transform(_z).fillna(0)
    # 地元
    df["p_local"] = (df["pref"].astype(str) == df["venue"].map(VENUE_PREF)).astype(float)
    return df.drop(columns=["_hb"])


P_FEATURES = ["p_s", "p_b", "p_head_b", "p_jiriki_head", "p_oikomi_2nd",
              "p_banate", "p_agari", "p_form", "p_local"]
P_LABEL = dict(p_s="S（スタート）", p_b="B（バック）", p_head_b="前の先行力（番手から）",
               p_jiriki_head="自力型×先頭", p_oikomi_2nd="差し型×番手", p_banate="番手巧者",
               p_agari="上がりタイム", p_form="調子（直近の着順）", p_local="地元")
