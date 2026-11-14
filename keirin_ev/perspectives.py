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
  ── 独自の視点（2026/11 追加） ──
  今場所の調子   : 同じ開催の前日までの着順（場所の中での上り調子・下り調子）
  ギヤ変更       : 前に出走したときからのギヤ倍数の変化（上げた選手は勝負気配）
  先行争い       : 他ラインの先頭の B（主導権争いが激しいほど先頭は苦しく、番手・まくりに展開が向く）
  ラインの厚み   : 先頭から見た、後ろの選手の競走得点（強い番手を連れた先行は残りやすい）
  番手の格上度   : 番手の競走得点 − 前の選手の競走得点（番手が格上なら差しやすい）
  今日のバンク   : 同じ場・同じ日のそれまでのレースで、先頭・番手のどちらが勝っているか
                   （当日の風・バンク状態の傾向。レース後に取り込む当日結果を使う）

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
    df = add_original(df, history)
    return df.drop(columns=["_hb"])


# ---------- 独自の視点 ----------
TODAY_K = 4           # 今日のバンク傾向の縮め具合（この数のレースぶん平均に寄せる）


@lru_cache(maxsize=4)
def _gear_table(dates_key):
    """選手ごと・日付ごとのギヤ倍数（前回出走のギヤを引くため）"""
    from . import store
    E = []
    for d in dates_key:
        e = store.read(d, "entries")
        if len(e) and "gear" in e:
            E.append(e.assign(date=d)[["name", "pref", "term", "date", "gear"]])
    if not E:
        return pd.DataFrame(columns=["rider", "date", "prev_gear"])
    e = pd.concat(E, ignore_index=True)
    e["rider"] = rider_key(e)
    e["gear"] = pd.to_numeric(e["gear"], errors="coerce")
    g = e.dropna(subset=["gear"]).groupby(["rider", "date"])["gear"].last().reset_index()
    g["date"] = pd.to_datetime(g["date"])
    return g.rename(columns={"gear": "prev_gear"}).sort_values("date")


def gear_change(df):
    from . import store
    out = pd.Series(0.0, index=df.index)
    if "gear" not in df or "name" not in df:
        return out
    tab = _gear_table(tuple(store.collected_dates()))
    if tab.empty:
        return out
    k = pd.DataFrame({"rider": rider_key(df), "date": pd.to_datetime(df["date"]),
                      "gear": pd.to_numeric(df["gear"], errors="coerce"), "_i": df.index}).dropna(subset=["date"])
    m = pd.merge_asof(k.sort_values("date"), tab, on="date", by="rider",
                      allow_exact_matches=False, direction="backward")
    v = (m["gear"] - m["prev_gear"]).fillna(0).clip(-0.2, 0.2) * 10
    out.loc[m["_i"].values] = v.values
    return out


def meet_form(history):
    """今場所（which=今）の着順の平均"""
    if history is None or history.empty or "which" not in history:
        return pd.DataFrame(columns=["race_id", "car", "meet_avg"])
    h = history[(history["which"] == "今")].dropna(subset=["rank"])
    h = h[h["rank"] <= 9]
    a = h.assign(race_id=h["race_id"].astype(str)).groupby(["race_id", "car"])["rank"].mean()
    return a.rename("meet_avg").reset_index()


_TODAY = {}


def today_winners(date):
    """その日の各レースの勝者の位置（race_id, venue, race_no, w_head, w_2nd）
    確定データ（data/daily）が無ければ、当日運用で取り込んだ結果（data/live/日付/today.csv）を使う"""
    from . import store
    from pathlib import Path
    if date in _TODAY:
        return _TODAY[date]
    out = pd.DataFrame(columns=["race_id", "venue", "race_no", "w_head", "w_2nd"])
    res, ent, rc = store.read(date, "results"), store.read(date, "entries"), store.read(date, "races")
    if len(res) and len(ent) and len(rc):
        w = res[["race_id", "first"]].merge(ent[["race_id", "car", "line_pos", "line_size"]],
                                           left_on=["race_id", "first"], right_on=["race_id", "car"])
        w = w.merge(rc[["race_id", "venue", "race_no"]], on="race_id")
        w["w_head"] = ((w["line_pos"] == 1) & (w["line_size"] > 1)).astype(float)
        w["w_2nd"] = (w["line_pos"] == 2).astype(float)
        out = w[out.columns]
    else:
        f = Path("data/live") / date / "today.csv"
        if f.exists():
            t = pd.read_csv(f, dtype={"race_id": str}).drop_duplicates("race_id", keep="last")
            out = t[out.columns]
    out = out.assign(race_id=out["race_id"].astype(str))
    if date < pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d"):
        _TODAY[date] = out                      # 当日分は増えていくのでキャッシュしない
    return out


def today_bias(df):
    """同じ場・同じ日のそれまでのレースの「先頭の勝ち」「番手の勝ち」の偏り（全体平均との差）"""
    zero = pd.DataFrame({"tb_head": 0.0, "tb_2nd": 0.0}, index=df.index)
    if "date" not in df or df["date"].isna().all():
        return zero
    rno = df["race_id"].astype(str).str[-4:].astype(int)
    out = zero.copy()
    BASE_H, BASE_2 = 0.48, 0.36                 # 先頭・番手が勝つ割合（全体）
    for (d, v), idx in df.groupby(["date", "venue"]).groups.items():
        w = today_winners(str(d))
        w = w[w["venue"] == v]
        if w.empty:
            continue
        for r in rno.loc[idx].unique():
            p = w[w["race_no"] < r]
            n = len(p)
            if n == 0:
                continue
            ii = rno.loc[idx][rno.loc[idx] == r].index
            out.loc[ii, "tb_head"] = (p["w_head"].sum() + TODAY_K * BASE_H) / (n + TODAY_K) - BASE_H
            out.loc[ii, "tb_2nd"] = (p["w_2nd"].sum() + TODAY_K * BASE_2) / (n + TODAY_K) - BASE_2
    return out


def add_original(df, history):
    g = df.groupby("race_id")
    is_head = (df["line_pos"] == 1) & (df["line_size"].fillna(1) > 1)
    is_2nd = df["line_pos"] == 2
    # 今場所の調子
    mf = meet_form(history)
    df = df.merge(mf, on=["race_id", "car"], how="left")
    df["p_meet"] = -df.groupby("race_id")["meet_avg"].transform(_z).fillna(0)
    # ギヤ変更
    df["p_gear"] = gear_change(df)
    # 先行争い：自分以外のライン先頭（単騎の逃げ型も含む）の B の最大
    sc = pd.to_numeric(df["score"], errors="coerce")
    df["_zs"] = sc.groupby(df["race_id"]).transform(_z).fillna(0)
    lead = is_head | ((df["line_size"].fillna(1) <= 1) & df["style"].fillna("").str.contains("逃"))
    hb = df["p_b"].where(lead)
    mx1 = hb.groupby(df["race_id"]).transform("max")
    # 自分が最大のときは2番目を使う
    def second_max(s):
        v = s.dropna().sort_values(ascending=False).values
        return v[1] if len(v) > 1 else np.nan
    mx2 = hb.groupby(df["race_id"]).transform(second_max)
    rival = np.where(lead & (hb == mx1), mx2, mx1)
    rival = pd.Series(rival, index=df.index).fillna(-1.0)
    df["p_head_rival"] = np.where(is_head, rival, 0.0)
    df["p_2nd_rival"] = np.where(is_2nd, rival, 0.0)
    # ラインの厚み（先頭から見た後ろの得点）
    back = df[df["line_pos"] >= 2].groupby(["race_id", "line_id"])["_zs"].mean().rename("_back").reset_index()
    df = df.merge(back, on=["race_id", "line_id"], how="left")
    df["p_line_back"] = np.where(is_head.values, df["_back"].fillna(0), 0.0)
    # 番手の格上度
    hz = df[df["line_pos"] == 1][["race_id", "line_id", "_zs"]].rename(columns={"_zs": "_hz"})
    df = df.merge(hz.drop_duplicates(["race_id", "line_id"]), on=["race_id", "line_id"], how="left")
    df["p_2nd_gap"] = np.where(is_2nd.values, (df["_zs"] - df["_hz"]).fillna(0), 0.0)
    # 今日のバンク
    tb = today_bias(df)
    df["p_today_head"] = np.where(is_head.values, tb["tb_head"].values * 10, 0.0)
    df["p_today_2nd"] = np.where(is_2nd.values, tb["tb_2nd"].values * 10, 0.0)
    return df.drop(columns=["_zs", "_back", "_hz"])


ORIG_FEATURES = ["p_meet", "p_gear", "p_head_rival", "p_2nd_rival", "p_line_back",
                 "p_2nd_gap", "p_today_head", "p_today_2nd"]

P_FEATURES = ["p_s", "p_b", "p_head_b", "p_jiriki_head", "p_oikomi_2nd",
              "p_banate", "p_agari", "p_form", "p_local"] + ORIG_FEATURES
P_LABEL = dict(p_s="S（スタート）", p_b="B（バック）", p_head_b="前の先行力（番手から）",
               p_jiriki_head="自力型×先頭", p_oikomi_2nd="差し型×番手", p_banate="番手巧者",
               p_agari="上がりタイム", p_form="調子（直近の着順）", p_local="地元",
               p_meet="今場所の調子", p_gear="ギヤ変更", p_head_rival="先行争い（先頭）",
               p_2nd_rival="先行争い（番手）", p_line_back="ラインの厚み", p_2nd_gap="番手の格上度",
               p_today_head="今日のバンク（先頭）", p_today_2nd="今日のバンク（番手）")
