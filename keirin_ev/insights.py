"""独自の視点の解説データ（サイトの「解説」と、レースごとの一言解説）

すべて「実際の結果」と「市場（オッズ）の評価」の差で見る。
市場の評価は3連単オッズから逆算した確率（1着確率・3着内確率）。
"""
import numpy as np
import pandas as pd
from . import store

POS_ORDER = ["先頭", "番手", "3番手", "4番手以降", "単騎"]


def position(line_pos, line_size):
    if line_size is None or line_size != line_size or line_size <= 1:
        return "単騎"
    return {1: "先頭", 2: "番手", 3: "3番手"}.get(int(line_pos), "4番手以降")


def market_marginals(odds):
    """3連単オッズ → 各選手の 市場1着確率・市場3着内確率"""
    o = odds.copy()
    o["q"] = (1 / o["odds"]) / (1 / o["odds"]).groupby(o["race_id"]).transform("sum")
    cs = o["combo"].str.split("-", expand=True).astype(int)
    w = o.assign(car=cs[0]).groupby(["race_id", "car"])["q"].sum().rename("mkt_win")
    t = pd.concat([o.assign(car=cs[i])[["race_id", "car", "q"]] for i in range(3)])
    t = t.groupby(["race_id", "car"])["q"].sum().rename("mkt_top3")
    return pd.concat([w, t], axis=1).fillna(0).reset_index()


def load_all():
    E, R, O, F, H = [], [], [], [], []
    for d in store.collected_dates():
        E.append(store.read(d, "entries").assign(date=d))
        R.append(store.read(d, "results")); O.append(store.read(d, "odds_trifecta"))
        F.append(store.read(d, "finish")); H.append(store.read(d, "history"))
    cat = lambda x: pd.concat([y for y in x if len(y)], ignore_index=True) if any(len(y) for y in x) else pd.DataFrame()
    return cat(E), cat(R), cat(O), cat(F), cat(H)


def rider_table():
    e, r, o, f, h = load_all()
    if e.empty or r.empty:
        return None
    e = e[e["class"] != "L1"]
    x = e.merge(r[["race_id", "first", "second", "third"]], on="race_id")
    x = x.merge(market_marginals(o), on=["race_id", "car"], how="left")
    x["win"] = x["car"] == x["first"]
    x["top3"] = x["car"].isin([]) | (x["car"] == x["first"]) | (x["car"] == x["second"]) | (x["car"] == x["third"])
    x["位置"] = [position(p, z) for p, z in zip(x["line_pos"], x["line_size"])]
    x["rider"] = x["name"].astype(str) + "|" + x["pref"].astype(str) + "|" + x["term"].astype(str)
    return x


def position_stats(x):
    g = x.groupby("位置").agg(n=("car", "size"), win=("win", "mean"), mkt_win=("mkt_win", "mean"),
                              top3=("top3", "mean"), mkt_top3=("mkt_top3", "mean")).reindex(POS_ORDER).dropna()
    # 差の標準誤差（ざっくり：二項分布）
    g["se_top3"] = np.sqrt(g["top3"] * (1 - g["top3"]) / g["n"])
    g["se_win"] = np.sqrt(g["win"] * (1 - g["win"]) / g["n"])
    return [dict(pos=k, n=int(v.n), win=v.win, mkt_win=v.mkt_win, top3=v.top3, mkt_top3=v.mkt_top3,
                 se_top3=v.se_top3, se_win=v.se_win) for k, v in g.iterrows()]


def banate_ranking(x, min_n=3, k=8, top=10):
    """番手巧者：番手のときの3着内が、市場の評価をどれだけ上回ったか（回数で縮めた値）"""
    b = x[x["位置"] == "番手"]
    g = b.groupby("rider").agg(n=("car", "size"), top3=("top3", "sum"), exp=("mkt_top3", "sum"),
                               score=("score", "last"), cls=("class", "last"))
    g = g[g["n"] >= min_n]
    g["plus"] = (g["top3"] - g["exp"]) / (g["n"] + k)     # 1走あたりの上積み（少ない回数は0に寄せる）
    g = g.sort_values("plus", ascending=False).head(top)
    out = []
    for rid, v in g.iterrows():
        name, pref, term = rid.split("|")
        out.append(dict(name=name, pref=pref, term=term.replace(".0", ""), cls=v.cls, n=int(v.n),
                        top3=int(v.top3), exp=round(float(v.exp), 2), plus=float(v.plus)))
    return out


def agari_index(history, race_id_cars=None, last_n=6):
    """直近の上がりタイム（最後の200m）の平均。小さいほど脚がある"""
    if history is None or history.empty:
        return pd.DataFrame(columns=["race_id", "car", "agari_avg", "agari_n"])
    h = history.dropna(subset=["agari"]).sort_values("date")
    h = h[(h["agari"] > 8) & (h["agari"] < 16)]
    g = h.groupby(["race_id", "car"]).tail(last_n).groupby(["race_id", "car"])["agari"]
    return pd.DataFrame({"agari_avg": g.mean(), "agari_n": g.size()}).reset_index()


def agari_stats(x, h):
    """上がり順位（レース内で上がり平均が速い順）別の成績と市場評価"""
    a = agari_index(h)
    if a.empty:
        return None
    y = x.merge(a, on=["race_id", "car"])
    if y.empty:
        return None
    y["agari_rank"] = y.groupby("race_id")["agari_avg"].rank(method="first")
    y["band"] = pd.cut(y["agari_rank"], [0, 1, 2, 3, 9], labels=["最速", "2位", "3位", "4位以下"])
    g = y.groupby("band", observed=True).agg(n=("car", "size"), win=("win", "mean"), mkt_win=("mkt_win", "mean"),
                                             top3=("top3", "mean"), mkt_top3=("mkt_top3", "mean"))
    return [dict(band=str(k), n=int(v.n), win=v.win, mkt_win=v.mkt_win, top3=v.top3, mkt_top3=v.mkt_top3)
            for k, v in g.iterrows()]


def build():
    x = rider_table()
    if x is None:
        return None
    _, _, _, _, h = load_all()
    return dict(n_races=int(x["race_id"].nunique()), n_entries=int(len(x)),
                period=f"{x['date'].min()}〜{x['date'].max()}",
                position=position_stats(x), banate=banate_ranking(x), agari=agari_stats(x, h))


# ---------- レースごとの一言解説（当日運用で使う） ----------
def race_notes(P, T, ent, his, odds, date=None, venue=None):
    """P: このレースの3連単確率(504,)  T: RaceTensor(1レース)  ent: 出走表  his: 前場所までの成績"""
    from .model import COMBOS, MAXN
    notes = []
    cars = T.cars[0]
    k = int(T.mask[0].sum())
    p1 = np.zeros(MAXN); p3 = np.zeros(MAXN)
    for n, (i, j, l) in enumerate(COMBOS):
        if P[n] > 0:
            p1[i] += P[n]; p3[i] += P[n]; p3[j] += P[n]; p3[l] += P[n]
    m = market_marginals(odds)
    mk = m.set_index("car")
    e = ent.set_index("car")
    # ラインの1着確率
    lines = {}
    for idx in range(k):
        c = int(cars[idx]); lid = e.loc[c, "line_id"]
        lines.setdefault(lid, []).append((e.loc[c, "line_pos"], c, p1[idx]))
    lp = sorted([(sum(v[2] for v in vs), [v[1] for v in sorted(vs)]) for vs in lines.values()], reverse=True)
    line_probs = [dict(cars=c, p=float(p)) for p, c in lp]
    if lp:
        notes.append(f"1着が出るライン：{'-'.join(map(str, lp[0][1]))} が {lp[0][0]:.0%}")
    # 番手の評価差
    for idx in range(k):
        c = int(cars[idx])
        if position(e.loc[c, "line_pos"], e.loc[c, "line_size"]) == "番手" and c in mk.index:
            d = p3[idx] - mk.loc[c, "mkt_top3"]
            if d >= 0.05:
                notes.append(f"番手 {c} {e.loc[c, 'name']}：3着内 {p3[idx]:.0%}（市場 {mk.loc[c, 'mkt_top3']:.0%}）で過小評価")
    # 上がり最速
    a = agari_index(his)
    if len(a) >= 2:
        a = a.sort_values("agari_avg")
        c = int(a.iloc[0]["car"]); gap = a.iloc[1]["agari_avg"] - a.iloc[0]["agari_avg"]
        notes.append(f"上がり最速：{c} {e.loc[c, 'name'] if c in e.index else ''}（直近平均 {a.iloc[0]['agari_avg']:.1f}秒、2位と{gap:.1f}秒差）")
    # いろんな視点（S・B・番手巧者・調子・地元）
    try:
        notes += view_notes(ent, his, date, venue, p1, p3, cars, k)
    except Exception as ex:
        print("視点の解説失敗:", ex, flush=True)
    # 単騎の過大評価
    for idx in range(k):
        c = int(cars[idx])
        if position(e.loc[c, "line_pos"], e.loc[c, "line_size"]) == "単騎" and c in mk.index:
            d = p1[idx] - mk.loc[c, "mkt_win"]
            if d <= -0.04:
                notes.append(f"単騎 {c} {e.loc[c, 'name']}：1着は市場ほど高くない（{p1[idx]:.0%}／市場 {mk.loc[c, 'mkt_win']:.0%}）")
    return notes, line_probs


def view_notes(ent, his, date, venue, p1, p3, cars, k):
    """S・B・前の先行力・番手巧者・調子・地元 の一言"""
    from .perspectives import rider_key, banate_skill, history_stats, VENUE_PREF
    e = ent.copy()
    for c in ("s_cnt", "back_cnt"):
        e[c] = pd.to_numeric(e.get(c), errors="coerce")
    e = e.set_index("car")
    nm = lambda c: f"{c} {e.loc[c, 'name']}"
    idx_of = {int(cars[i]): i for i in range(k)}
    out = []
    s = e["s_cnt"].dropna().sort_values(ascending=False)
    if len(s) >= 2 and s.iloc[0] >= 3 and s.iloc[0] - s.iloc[1] >= 2:
        out.append(f"S（スタート）は {nm(s.index[0])} が取りやすい（S {int(s.iloc[0])}回、次点 {int(s.iloc[1])}回）")
    b = e["back_cnt"].dropna().sort_values(ascending=False)
    if len(b) >= 2 and b.iloc[0] >= 3:
        c = int(b.index[0]); msg = f"B（バック）最多 {nm(c)}（{int(b.iloc[0])}回）＝主導権を取りやすい"
        if e.loc[c, "line_pos"] == 1 and e.loc[c, "line_size"] > 1:
            fol = e[(e["line_id"] == e.loc[c, "line_id"]) & (e["line_pos"] == 2)]
            if len(fol):
                f = int(fol.index[0]); msg += f"。番手 {nm(f)} に展開が向く"
                if f in idx_of:
                    msg += f"（3着内 {p3[idx_of[f]]:.0%}）"
        out.append(msg)
    if date and {"name", "pref"} <= set(e.columns):
        x = e.reset_index()
        x["banate_plus"] = banate_skill(pd.DataFrame({"rider": rider_key(x), "date": date}, index=x.index))
        for r in x[(x["line_pos"] == 2) & (x["banate_plus"] >= 0.03)].itertuples():
            out.append(f"番手巧者 {r.car} {r.name}：番手で市場の見積もりより1走あたり+{r.banate_plus:.2f}回 3着内に来ている")
    hs = history_stats(his)
    if len(hs) >= 3:
        hs = hs.dropna(subset=["form_avg"]).sort_values("form_avg")
        if len(hs) >= 2 and hs.iloc[1]["form_avg"] - hs.iloc[0]["form_avg"] >= 0.8:
            c = int(hs.iloc[0]["car"])
            if c in e.index:
                out.append(f"調子が良い {nm(c)}（直近の平均 {hs.iloc[0]['form_avg']:.1f}着）")
    if venue in VENUE_PREF:
        loc = [int(c) for c in e.index if str(e.loc[c, "pref"]) == VENUE_PREF[venue]]
        if loc:
            out.append("地元：" + "・".join(nm(c) for c in loc))
    return out
