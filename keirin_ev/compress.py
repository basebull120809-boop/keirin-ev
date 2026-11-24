"""いろんな情報を集めて「圧縮」し、市場が見落としている分を学習する（研究用）

1. 集める   : 自前の着順データから、そのレースより前の成績だけで選手ごとの指標を作る
              （天候別・バンク周長別・場別・位置別の3着内率、S率・B率、決まり手の割合、上がり、
               直近の着順、そして「過去に市場の見積もりをどれだけ上回ったか」）
              ＋ 出走表（競走得点・ライン・記者予想など）
2. 圧縮する : 勾配ブースティング（LightGBM）で、市場の1着確率・3着内確率を出発点にして
              「市場からのずれ」だけを学習する（= 100項目前後を1つの補正値に圧縮）
3. 判定する : 学習に使っていない期間で、市場より当たりの見積もりが良くなるか（対数尤度）と、
              補正した確率で割安な目だけ買ったときの回収率を見る

使い方: python main.py compress [検証開始日]
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from .store import collected_dates, read

SMOOTH = 8.0          # 率をならす強さ（出走数が少ない選手は全体平均に寄せる）

# 全体平均など「データ全体から決める定数」。学習時に記録し、当日判定（一部の選手だけで計算）ではその値を使う
_G = {}
_FROZEN = [False]
USE_CR = False


def gm(name, v):
    if _FROZEN[0] and name in _G:
        return _G[name]
    _G[name] = float(v)
    return _G[name]


# ---------- 読み込み ----------
def load_all(log=print):
    days = collected_dates()
    T = {t: [] for t in ("races", "entries", "finish", "results", "odds_trifecta",
                          "odds_exacta", "odds_quinella", "odds_wide")}
    for d in days:
        for t in T:
            df = read(d, t)
            if len(df):
                T[t].append(df)
    T = {t: pd.concat(v, ignore_index=True) for t, v in T.items()}
    keys = dict(races=["race_id"], results=["race_id"], entries=["race_id", "car"], finish=["race_id", "car"],
                odds_trifecta=["race_id", "combo", "snapshot"], odds_exacta=["race_id", "combo"],
                odds_quinella=["race_id", "combo"], odds_wide=["race_id", "combo"])
    for t in T:
        T[t]["race_id"] = T[t]["race_id"].astype(str)
        k = [c for c in keys[t] if c in T[t]]
        T[t] = T[t].drop_duplicates(k, keep="last").reset_index(drop=True)   # 同じレースが2日分のファイルに入っている場合
    log(f"読み込み: {len(days)}日 レース{T['races'].race_id.nunique()} 出走{len(T['entries'])}")
    return T


def market(odds):
    o = odds[odds["snapshot"] == "final"] if "snapshot" in odds and (odds["snapshot"] == "final").any() else odds
    o = o[o.odds > 0].copy()
    o["inv"] = 1 / o.odds
    o["q"] = o.inv / o.groupby("race_id").inv.transform("sum")
    p = o.combo.str.split("-", expand=True).astype(int)
    o["c1"], o["c2"], o["c3"] = p[0], p[1], p[2]
    win = o.groupby(["race_id", "c1"]).q.sum().rename("m_win")
    top3 = pd.concat([o.groupby(["race_id", c]).q.sum() for c in ("c1", "c2", "c3")], axis=1).sum(axis=1)
    m = pd.concat([win, top3.rename("m_top3")], axis=1).reset_index()
    m.columns = ["race_id", "car", "m_win", "m_top3"]
    return o[["race_id", "combo", "c1", "c2", "c3", "odds", "q"]], m


# ---------- 1. 集める（その日より前だけ） ----------
def _prior(df, key, cols):
    """key（選手）ごとに、日付順で「前の日まで」の累計を返す（同じ日の分は含めない）"""
    g = df.groupby(key + ["date"], sort=False)[cols].sum()
    g = g.groupby(level=list(range(len(key)))).cumsum() - g     # その日の分を引く = 前日までの累計
    return df[key + ["date"]].merge(g.reset_index(), on=key + ["date"], how="left")[cols].values


def prepare(races, entries, finish, mk):
    """出走表・着順・市場の確率を1つの表にし、着順から作る素の列を足す（日付順）"""
    r = races[["race_id", "date", "venue", "race_no", "grade", "bank_len", "wind", "weather"]]
    e = entries.merge(r, on="race_id", how="inner")
    f = finish[["race_id", "car", "rank", "kimarite", "sb", "agari"]]
    e = e.merge(f, on=["race_id", "car"], how="left").merge(mk, on=["race_id", "car"], how="left")
    e = e[e.m_win.notna()].copy()
    e["rid"] = e["name"].astype(str) + "_" + e["term"].astype(str)
    e = e.sort_values(["date", "race_no"]).reset_index(drop=True)
    rk = e["rank"]
    e["y1"] = (rk == 1).astype(float)
    e["y2"] = (rk <= 2).astype(float)
    e["y3"] = (rk <= 3).astype(float)
    e["fin"] = rk.notna().astype(float)
    sb = e["sb"].fillna("").astype(str)
    e["gotS"] = sb.str.contains("S").astype(float)
    e["gotB"] = sb.str.contains("B").astype(float)
    km = e["kimarite"].fillna("").astype(str)
    for k, c in (("逃", "k_nige"), ("捲", "k_makuri"), ("差", "k_sashi"), ("ク", "k_mark")):
        e[c] = km.str.contains(k).astype(float)
    e["ag"] = pd.to_numeric(e["agari"], errors="coerce")
    e["ag_n"] = e["ag"].notna().astype(float)
    e["ag"] = e["ag"].fillna(0)
    e["over1"] = e["y1"] - e["m_win"]                  # 市場の見積もりを上回った分
    e["over3"] = e["y3"] - e["m_top3"]
    e["rkv"] = rk.fillna(9).clip(upper=9)
    e["one"] = 1.0
    e["pos"] = np.select([e.line_size.fillna(1) <= 1, e.line_pos == 1, e.line_pos == 2], ["single", "head", "2nd"], "3rd")
    e["bank"] = np.select([e.bank_len <= 335, e.bank_len >= 500], ["333", "500"], "400")
    e["wx"] = e["weather"].fillna("?").astype(str).str[:1]
    return e


def rider_stats(T, mk, log=print):
    e = prepare(T["races"], T["entries"], T["finish"], mk)
    X = stats(e)
    log(f"選手の指標: {X.shape[1]}項目")
    return e, X


def stats(e):
    base = ["one", "fin", "y1", "y2", "y3", "gotS", "gotB", "k_nige", "k_makuri", "k_sashi", "k_mark",
            "ag", "ag_n", "over1", "over3", "rkv"]
    P = pd.DataFrame(_prior(e, ["rid"], base), columns=["c_" + c for c in base], index=e.index)
    n = P["c_one"]
    g1, g3 = gm("y1", e.y1.mean()), gm("y3", e.y3.mean())
    X = pd.DataFrame(index=e.index)
    X["n_prior"] = n
    X["r1"] = (P.c_y1 + SMOOTH * g1) / (n + SMOOTH)
    X["r2"] = (P.c_y2 + SMOOTH * gm("y2", e.y2.mean())) / (n + SMOOTH)
    X["r3"] = (P.c_y3 + SMOOTH * g3) / (n + SMOOTH)
    X["s_rate"] = (P.c_gotS + SMOOTH * gm("gotS", e.gotS.mean())) / (n + SMOOTH)
    X["b_rate"] = (P.c_gotB + SMOOTH * gm("gotB", e.gotB.mean())) / (n + SMOOTH)
    for c in ("k_nige", "k_makuri", "k_sashi", "k_mark"):
        X[c] = (P["c_" + c] + SMOOTH * gm(c, e[c].mean())) / (n + SMOOTH)
    X["agari"] = np.where(P.c_ag_n > 0, P.c_ag / P.c_ag_n.clip(lower=1), np.nan)
    X["over1"] = P.c_over1 / (n + SMOOTH)               # 過去に市場を上回った度合い（少数はゼロに寄せる）
    X["over3"] = P.c_over3 / (n + SMOOTH)
    X["avg_rank"] = (P.c_rkv + SMOOTH * 4.5) / (n + SMOOTH)

    # 直近10走（前の日まで）
    e["_i"] = e.groupby("rid").cumcount()
    for c, nm in (("y3", "r3_last10"), ("rkv", "rank_last10"), ("over3", "over3_last10")):
        roll = e.groupby("rid")[c].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
        X[nm] = roll
    # 同じ日に2走していると shift(1) が同日を含むので、日付単位で前日までに揃える
    last = e.groupby("rid")["date"].shift(1)
    X["days_since"] = (pd.to_datetime(e.date) - pd.to_datetime(last)).dt.days

    # 条件別の3着内率（今回と同じ条件の、前の日までの成績）
    for cond in ("pos", "bank", "wx", "venue"):
        Q = pd.DataFrame(_prior(e, ["rid", cond], ["one", "y3", "over3"]), index=e.index, columns=["n", "y3", "ov"])
        X[f"r3_{cond}"] = (Q.y3 + SMOOTH * X["r3"]) / (Q.n + SMOOTH)    # 全体の率に寄せる
        X[f"over3_{cond}"] = Q.ov / (Q.n + SMOOTH)
        X[f"n_{cond}"] = Q.n
    X = X.join(head_to_head(e))
    X = X.join(line_dynamics(e, X))
    return X


def line_dynamics(e, X):
    """レース展開の数字（どれも前の日までの成績から）

    先頭として : ラインから1着が出た割合（主導権を取って決めた／番手を勝たせた）、自分が3着内に残った割合
    番手として : 前（先頭）に先着した割合、前とワンツーを決めた割合
    この2人で  : 同じ先頭・番手の組み合わせで走った回数と、ワンツー・2人とも3着内の割合
    レース内    : 逃げ型の人数（先行争いの激しさ）、ラインの数、自分のラインの強さの順位
    """
    d = e[["race_id", "date", "rid", "line_id", "line_pos", "line_size", "rkv"]].copy()
    d["lk"] = d.race_id + "_" + d.line_id.astype(str)
    multi = d.line_size.fillna(1) > 1
    head = d[multi & (d.line_pos == 1)][["lk", "rid", "rkv"]].rename(columns={"rid": "h_rid", "rkv": "h_rk"})
    sec = d[multi & (d.line_pos == 2)][["lk", "rid", "rkv"]].rename(columns={"rid": "s_rid", "rkv": "s_rk"})
    best = d.groupby("lk").rkv.min().rename("best")
    d = d.merge(head, on="lk", how="left").merge(sec, on="lk", how="left").merge(best, on="lk", how="left")
    d.index = e.index
    one = pd.Series(1.0, index=d.index)
    D = pd.DataFrame(index=e.index)

    # 先頭として
    ish = (multi & (d.line_pos == 1)).astype(float)
    t = pd.DataFrame(dict(rid=d.rid, date=d.date, n=ish, won=ish * (d.best == 1), top3=ish * (d.rkv <= 3)))
    P = pd.DataFrame(_prior(t, ["rid"], ["n", "won", "top3"]), index=e.index, columns=["n", "won", "top3"])
    gw = gm("head_won", t.won.sum() / max(t.n.sum(), 1)); g3 = gm("head_top3", t.top3.sum() / max(t.n.sum(), 1))
    D["head_n"] = P.n
    D["head_linewin"] = (P.won + 8 * gw) / (P.n + 8)
    D["head_top3"] = (P.top3 + 8 * g3) / (P.n + 8)

    # 番手として
    iss = (multi & (d.line_pos == 2)).astype(float)
    t = pd.DataFrame(dict(rid=d.rid, date=d.date, n=iss, beat=iss * (d.rkv < d.h_rk),
                          onetwo=iss * ((d.rkv <= 2) & (d.h_rk <= 2))))
    P = pd.DataFrame(_prior(t, ["rid"], ["n", "beat", "onetwo"]), index=e.index, columns=["n", "beat", "onetwo"])
    gb = gm("sec_beat", t.beat.sum() / max(t.n.sum(), 1)); g12 = gm("sec_onetwo", t.onetwo.sum() / max(t.n.sum(), 1))
    D["sec_n"] = P.n
    D["sec_beat_head"] = (P.beat + 8 * gb) / (P.n + 8)
    D["sec_onetwo"] = (P.onetwo + 8 * g12) / (P.n + 8)

    # 先頭・番手の組み合わせ（番手の行に付ける。先頭の行には相方の値を付ける）
    pair = d[multi & (d.line_pos == 2)].copy()
    pair["n"] = 1.0
    pair["onetwo"] = ((pair.rkv <= 2) & (pair.h_rk <= 2)).astype(float)
    pair["both3"] = ((pair.rkv <= 3) & (pair.h_rk <= 3)).astype(float)
    pair = pair[pair.h_rid.notna()]
    Pp = pd.DataFrame(_prior(pair, ["h_rid", "rid"], ["n", "onetwo", "both3"]), index=pair.index,
                      columns=["n", "onetwo", "both3"])
    D["pair_n"] = np.nan
    D.loc[Pp.index, "pair_n"] = Pp.n
    D.loc[Pp.index, "pair_onetwo"] = (Pp.onetwo + 4 * g12) / (Pp.n + 4)
    D.loc[Pp.index, "pair_both3"] = Pp.both3 / (Pp.n + 4)
    pv = D.loc[Pp.index, ["pair_n", "pair_onetwo", "pair_both3"]].assign(lk=d.loc[Pp.index, "lk"])
    hidx = d.index[multi & (d.line_pos == 1)]
    hv = d.loc[hidx, ["lk"]].reset_index().merge(pv, on="lk", how="left").set_index("index")
    for c in ("pair_n", "pair_onetwo", "pair_both3"):
        D.loc[hv.index, c] = hv[c]

    # 自分の前（先頭）の成績を、番手・3番手の行に付ける
    hs = pd.DataFrame(dict(lk=d.lk, linewin=D.head_linewin, htop3=D.head_top3))[ish == 1].drop_duplicates("lk")
    m = d[["lk"]].merge(hs, on="lk", how="left")
    D["front_linewin"] = np.where(d.line_pos.values >= 2, m.linewin.values, np.nan)
    D["front_top3"] = np.where(d.line_pos.values >= 2, m.htop3.values, np.nan)
    # 番手の「前を抜く力」を先頭の行に付ける（強い番手を連れていると先頭は残りにくい）
    ss = pd.DataFrame(dict(lk=d.lk, beat=D.sec_beat_head))[iss == 1].drop_duplicates("lk")
    m = d[["lk"]].merge(ss, on="lk", how="left")
    D["behind_beat"] = np.where(ish.values == 1, m.beat.values, np.nan)

    # レース内の状況
    nige = (X["k_nige"] > gm("k_nige_med", X["k_nige"].median()) * 1.5).astype(float)
    D["n_nige_race"] = nige.groupby(e.race_id).transform("sum")
    D["n_lines"] = d.groupby("race_id").lk.transform("nunique")
    ls = pd.Series(D.head_linewin.where(ish == 1), index=e.index).groupby(d.lk).transform("max")
    D["line_strength_rank"] = ls.groupby(e.race_id).rank(ascending=False, method="min")
    return D


def head_to_head(e):
    """対戦成績：今回の相手それぞれに、前の日までに先着した回数−先着された回数（相手ごとにならして平均）"""
    a = e[["race_id", "date", "rid", "rkv"]].reset_index()
    p = a.merge(a[["race_id", "rid", "rkv"]], on="race_id", suffixes=("", "_o"))
    p = p[p.rid != p.rid_o]
    p["w"] = (p.rkv < p.rkv_o).astype(float) - (p.rkv > p.rkv_o).astype(float)
    p["n"] = 1.0
    g = p.groupby(["rid", "rid_o", "date"])[["w", "n"]].sum()
    g = g.groupby(level=[0, 1]).cumsum() - g                    # 前の日までの対戦
    p = p.merge(g.rename(columns={"w": "pw", "n": "pn"}).reset_index(), on=["rid", "rid_o", "date"], how="left")
    p["score"] = p.pw / (p.pn + 3.0)
    p["met"] = (p.pn > 0).astype(float)
    r = p.groupby("index").agg(h2h=("score", "mean"), h2h_sum=("pw", "sum"), h2h_met=("met", "mean"))
    return r


# ---------- レースの中での相対化 ＋ 出走表 ----------
TIP = {"◎": 5, "○": 4, "▲": 3, "△": 2, "注": 1, "×": 1}


def race_features(e, X):
    F = X.copy()
    g = e["race_id"]
    rel = ["r1", "r3", "s_rate", "b_rate", "k_nige", "k_makuri", "k_sashi", "agari", "avg_rank",
           "r3_pos", "r3_bank", "r3_wx", "r3_venue", "over3", "r3_last10", "h2h"]
    for c in rel:
        s = F[c]
        F[c + "_z"] = (s - s.groupby(g).transform("mean")) / s.groupby(g).transform("std").replace(0, np.nan)
    for c in ("score", "win_rate", "ren_rate", "san_rate", "gear", "s_cnt", "back_cnt", "nige", "makuri",
              "sashi", "mark", "line_pos", "line_size", "evaluation", "kiai", "wind", "bank_len", "age"):
        F[c] = pd.to_numeric(e[c], errors="coerce")
    F["score_z"] = (F.score - F.score.groupby(g).transform("mean")) / F.score.groupby(g).transform("std").replace(0, np.nan)
    F["tip"] = e["tip"].map(TIP).fillna(0)
    F["is_head"] = ((e.line_pos == 1) & (e.line_size > 1)).astype(float)
    F["is_2nd"] = (e.line_pos == 2).astype(float)
    F["single"] = (e.line_size.fillna(1) <= 1).astype(float)
    F["style_nige"] = e["style"].fillna("").str.contains("逃").astype(float)
    F["n_riders"] = g.map(g.value_counts())
    # ラインの仲間：先頭の B率・番手から見た前の先行力
    key = e["race_id"] + "_" + e["line_id"].astype(str)
    head_b = pd.Series(np.where(e.line_pos == 1, X.b_rate, np.nan), index=e.index).groupby(key).transform("max")
    F["line_head_b"] = head_b
    F["line_score_sum"] = F.score.groupby(key).transform("sum")
    F["rival_head_b_max"] = (pd.Series(np.where(e.line_pos == 1, X.b_rate, np.nan), index=e.index)
                             .groupby(g).transform("max") - head_b.fillna(0))
    # チャリロトの記者予想（2026/11 の検証で効果なし。使うときは USE_CR=True）
    try:
        if not USE_CR:
            raise RuntimeError("使わない")
        from .chariloto import read_all, MARK
        cr = read_all()
        if len(cr):
            cr = cr.drop_duplicates(["race_id", "car"], keep="last")
            cr["cr_mark"] = cr["mark"].map(MARK).fillna(0)
            m = e[["race_id", "car"]].merge(cr[["race_id", "car", "cr_mark"]], on=["race_id", "car"], how="left")
            F["cr_mark"] = m["cr_mark"].values
            F["cr_vs_tip"] = F["cr_mark"] - F["tip"]          # 2社の記者の評価の食い違い
    except Exception:
        pass
    F["log_m_win"] = np.log(e.m_win.clip(1e-4))
    F["log_m_top3"] = np.log(e.m_top3.clip(1e-4))
    return F


# ---------- 2. 圧縮（市場からのずれだけ学習） ----------
def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_residual(F, y, base_p, tr, params=None):
    import lightgbm as lgb
    params = params or dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=400,
                            feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                            verbose=-1, seed=1)
    ds = lgb.Dataset(F[tr], y[tr], init_score=_logit(base_p[tr]), free_raw_data=False)
    return lgb.train(params, ds, num_boost_round=params.pop("rounds", 400))


def predict_residual(model, F, base_p, race_id, top=1):
    z = _logit(base_p) + model.predict(F, raw_score=True)
    p = 1 / (1 + np.exp(-z))
    s = pd.Series(p, index=F.index).groupby(race_id).transform("sum")
    return (p / s * top).clip(1e-5, 0.999)


# ---------- 3. 判定 ----------
def evaluate(T, e, F, valid_from, log=print, valid_to="9999"):
    rid = e["race_id"]
    tr = (e.date < valid_from).values & e["rank"].notna().values
    va = ((e.date >= valid_from) & (e.date < valid_to)).values
    # 学習期間のうち最後の1か月は、ブースティングの回数決めにだけ使う
    y1, y3 = e.y1.values, e.y3.values
    m1 = fit_residual(F, y1, e.m_win.values, tr)
    m3 = fit_residual(F, y3, e.m_top3.values, tr)
    p1 = predict_residual(m1, F, e.m_win.values, rid, 1)
    p3 = predict_residual(m3, F, e.m_top3.values, rid, 3).clip(upper=0.99)
    V = e.loc[va, ["race_id", "car", "date", "y1", "y3", "m_win", "m_top3"]].copy()
    V["p1"], V["p3"] = p1[va], p3[va]
    V = V[V.race_id.isin(e.loc[va & e["rank"].notna().values, "race_id"])]
    ll = lambda p, y: (y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    w = V[V.y1 == 1]
    res = dict(valid_from=valid_from, races=int(V.race_id.nunique()),
               ll_win_market=float(np.log(w.m_win).mean()), ll_win_model=float(np.log(w.p1).mean()),
               ll_top3_market=float(ll(V.m_top3.clip(1e-4, .999), V.y3)), ll_top3_model=float(ll(V.p3, V.y3)))
    res["gain_win_per_race"] = res["ll_win_model"] - res["ll_win_market"]
    log(f"検証 {valid_from}〜 {res['races']}R  1着の対数尤度 市場{res['ll_win_market']:.4f} → 補正後{res['ll_win_model']:.4f}"
        f"（差 {res['gain_win_per_race']:+.4f}）  3着内 市場{res['ll_top3_market']:.4f} → {res['ll_top3_model']:.4f}")

    # 重要だった項目
    imp = pd.Series(m1.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    imp3 = pd.Series(m3.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    res["importance_win"] = {k: round(float(v), 1) for k, v in imp.head(20).items()}
    res["importance_top3"] = {k: round(float(v), 1) for k, v in imp3.head(20).items()}
    log("1着の補正に効いた項目: " + " ".join(imp.head(12).index))

    # 補正した確率で3連単・2車単・2車複を組み直し、割安な目だけ買う
    res["bets"] = bet_sim(T, V, log)
    return res


def bet_sim(T, V, log=print):
    adj = V.set_index(["race_id", "car"])
    f1 = (adj.p1 / adj.m_win).to_dict()                 # 1着の補正倍率
    f3 = (adj.p3 / adj.m_top3).to_dict()                # 3着内の補正倍率
    races = set(V.race_id)
    res = T["results"].set_index("race_id")
    out = {}
    tri, _ = market(T["odds_trifecta"][T["odds_trifecta"].race_id.isin(races)])
    a = tri.race_id.values
    k1 = np.array([f1.get((r, c), 1.0) for r, c in zip(a, tri.c1)])
    k2 = np.array([f3.get((r, c), 1.0) for r, c in zip(a, tri.c2)])
    k3 = np.array([f3.get((r, c), 1.0) for r, c in zip(a, tri.c3)])
    tri["q2"] = tri.q * k1 * np.sqrt(k2 * k3)
    tri["q2"] = tri.q2 / tri.groupby("race_id").q2.transform("sum")
    win = res.reindex(tri.race_id)
    tri["hit"] = ((win["first"].values == tri.c1.values) & (win["second"].values == tri.c2.values)
                  & (win["third"].values == tri.c3.values))
    tri["pay"] = np.where(tri.hit, win["tri_payout"].values, 0)
    tri["ev"] = tri.q2 * tri.odds
    # 2車単・2車複は3連単の確率を足し上げる
    ex = tri.groupby(["race_id", "c1", "c2"]).q2.sum().reset_index()
    ex["combo"] = ex.c1.astype(str) + "-" + ex.c2.astype(str)
    eo = T["odds_exacta"]; eo = eo[eo.race_id.isin(races) & (eo.odds > 0)]
    ex = ex.merge(eo[["race_id", "combo", "odds"]], on=["race_id", "combo"])
    w2 = res.reindex(ex.race_id)
    ex["hit"] = (w2["first"].values == ex.c1.values) & (w2["second"].values == ex.c2.values)
    ex["pay"] = np.where(ex.hit, w2["ex_payout"].values, 0)
    ex["ev"] = ex.q2 * ex.odds
    qn = ex.assign(a=np.minimum(ex.c1, ex.c2), b=np.maximum(ex.c1, ex.c2)).groupby(["race_id", "a", "b"]).q2.sum().reset_index()
    qn["combo"] = qn.a.astype(str) + "=" + qn.b.astype(str)
    qo = T["odds_quinella"]; qo = qo[qo.race_id.isin(races) & (qo.odds > 0)]
    qn = qn.merge(qo[["race_id", "combo", "odds"]], on=["race_id", "combo"])
    w3 = res.reindex(qn.race_id)
    qn["hit"] = (w3["qn_combo"].values == qn.combo.values)
    qn["pay"] = np.where(qn.hit, w3["qn_payout"].values, 0)
    qn["ev"] = qn.q2 * qn.odds
    # ワイド：2人とも3着内に入る確率（3連単の確率から足し上げ）
    wd = []
    for a_, b_ in ((0, 1), (0, 2), (1, 2)):
        cs = [tri.c1, tri.c2, tri.c3]
        wd.append(pd.DataFrame(dict(race_id=tri.race_id, a=np.minimum(cs[a_], cs[b_]), b=np.maximum(cs[a_], cs[b_]), q2=tri.q2)))
    wd = pd.concat(wd).groupby(["race_id", "a", "b"]).q2.sum().reset_index()
    wd["combo"] = wd.a.astype(str) + "=" + wd.b.astype(str)
    wo = T.get("odds_wide")
    if wo is not None and len(wo):
        wo = wo[wo.race_id.isin(races) & (wo.odds > 0)]
        wd = wd.merge(wo[["race_id", "combo", "odds"]], on=["race_id", "combo"])
        wpay = {}
        for r_, cs_, ps_ in zip(res.index, res["wide_combos"], res["wide_payouts"]):
            if isinstance(cs_, str) and isinstance(ps_, str) and r_ in races:
                for c_, p_ in zip(cs_.split("|"), ps_.split("|")):
                    try:
                        wpay[(r_, "=".join(sorted(c_.split("="), key=int)))] = float(p_)
                    except ValueError:
                        pass
        wd["pay"] = [wpay.get((r_, c_), 0.0) for r_, c_ in zip(wd.race_id, wd.combo)]
        wd = wd[wd.race_id.isin({k[0] for k in wpay})]
        wd["ev"] = wd.q2 * wd.odds                         # ワイドの odds は下限側（実際の払戻は下限〜上限）
    else:
        wd = wd.iloc[:0].assign(pay=[], ev=[])
    for name, d in (("3連単", tri), ("2車単", ex), ("2車複", qn), ("ワイド", wd)):
        rows = []
        for th in (0.9, 1.0, 1.1, 1.2, 1.3, 1.5):
            b = d[d.ev >= th]
            if len(b) == 0:
                continue
            rr = b.groupby("race_id").pay.sum()
            rows.append(dict(ev_min=th, points=int(len(b)), races=int(b.race_id.nunique()),
                             race_hit=round(float((rr > 0).mean() * 100), 1),
                             roi=round(float(b.pay.sum() / (100 * len(b)) * 100), 1)))
        out[name] = rows
        log(f"  {name}: " + " / ".join(f"期待値{r['ev_min']}以上 {r['points']}点 {r['races']}R 的中{r['race_hit']}% 回収{r['roi']}%" for r in rows))
    return out


def run(valid_from="2026-08-01", out="reports/compress.json", log=print):
    T = load_all(log)
    tri, mk = market(T["odds_trifecta"])
    e, X = rider_stats(T, mk, log)
    F = race_features(e, X)
    res = evaluate(T, e, F, valid_from, log)
    res["n_features"] = int(F.shape[1])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    run(*(sys.argv[1:2] or []))


# ---------- 本番用：全データで学習して保存 ----------
MODEL_DIR = Path("models")


def train_all(log=print):
    """全期間で学習し、models/compress_win.txt・compress_top3.txt・compress_meta.json に保存"""
    _FROZEN[0] = False
    _G.clear()
    T = load_all(log)
    _, mk = market(T["odds_trifecta"])
    e, X = rider_stats(T, mk, log)
    F = race_features(e, X)
    tr = e["rank"].notna().values
    m1 = fit_residual(F, e.y1.values, e.m_win.values, tr)
    m3 = fit_residual(F, e.y3.values, e.m_top3.values, tr)
    MODEL_DIR.mkdir(exist_ok=True)
    m1.save_model(str(MODEL_DIR / "compress_win.txt"))
    m3.save_model(str(MODEL_DIR / "compress_top3.txt"))
    meta = dict(G=_G, features=list(F.columns), trained_at=pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M"),
                n_rows=int(tr.sum()), last_date=str(e.date.max()))
    (MODEL_DIR / "compress_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"学習: {meta['n_rows']}行（〜{meta['last_date']}）→ models/compress_*.txt")
    return meta


# ---------- 当日判定：締切前のオッズで補正した2車複の確率 ----------
class LivePredictor:
    """起動時に過去の全データを1回だけ読み、レースごとにその出走選手の分だけ指標を計算して予想する"""

    def __init__(self, log=print):
        import lightgbm as lgb
        meta = json.loads((MODEL_DIR / "compress_meta.json").read_text(encoding="utf-8"))
        self.features = meta["features"]
        _G.clear(); _G.update(meta["G"]); _FROZEN[0] = True
        self.m1 = lgb.Booster(model_file=str(MODEL_DIR / "compress_win.txt"))
        self.m3 = lgb.Booster(model_file=str(MODEL_DIR / "compress_top3.txt"))
        T = load_all(log)
        _, mk = market(T["odds_trifecta"])
        self.hist = prepare(T["races"], T["entries"], T["finish"], mk)
        log(f"圧縮モデル: 過去 {len(self.hist)} 行を読み込み（学習 {meta['trained_at']}）")

    def predict(self, race, ent, od):
        """race: dict(race_id, date, venue, race_no, grade, bank_len, wind, weather)
        ent: 出走表（kdreams.parse_race の entries）  od: 3連単オッズ（combo, odds）
        戻り値: 選手ごとの DataFrame（car, m_win, m_top3, p1, p3）"""
        rid = str(race["race_id"])
        tri, mk = market(od.assign(race_id=rid))
        races = pd.DataFrame([race]).assign(race_id=rid)
        ent = ent.assign(race_id=rid)
        fin = pd.DataFrame(dict(race_id=rid, car=ent["car"].values, rank=np.nan, kimarite=np.nan, sb=np.nan, agari=np.nan))
        new = prepare(races, ent, fin, mk)
        rids = set(new["rid"])
        past_races = set(self.hist.loc[self.hist.rid.isin(rids), "race_id"])
        sub = pd.concat([self.hist[self.hist.race_id.isin(past_races)], new], ignore_index=True)
        sub = sub.sort_values(["date", "race_no"], kind="stable").reset_index(drop=True)
        X = stats(sub)
        F = race_features(sub, X)
        mask = (sub.race_id == rid).values
        Fr = F.loc[mask].reindex(columns=self.features)
        er = sub.loc[mask]
        z1 = _logit(er.m_win.values) + self.m1.predict(Fr, raw_score=True)
        z3 = _logit(er.m_top3.values) + self.m3.predict(Fr, raw_score=True)
        p1 = 1 / (1 + np.exp(-z1)); p1 = p1 / p1.sum()
        p3 = 1 / (1 + np.exp(-z3)); p3 = np.clip(p3 / p3.sum() * 3, 1e-5, 0.99)
        out = pd.DataFrame(dict(car=er.car.astype(int).values, m_win=er.m_win.values, m_top3=er.m_top3.values,
                                p1=p1, p3=p3))
        return out, tri


def quinella_plan(pred, tri, qn_odds, ev_min=1.0, stake=100, odds_max=10.0):
    """補正した確率で2車複を組み直し、期待値 ev_min 以上・オッズ odds_max 倍以下の目を返す（紙上運用 cmp2）
    odds_max は 2026-11-24 追加：上限がなく数百〜9999.9倍の目まで拾っていたため、10倍で切る"""
    f1 = dict(zip(pred.car, pred.p1 / pred.m_win))
    f3 = dict(zip(pred.car, pred.p3 / pred.m_top3))
    k = tri.c1.map(f1).fillna(1) * np.sqrt(tri.c2.map(f3).fillna(1) * tri.c3.map(f3).fillna(1))
    q2 = tri.q * k
    q2 = q2 / q2.sum()
    a = np.minimum(tri.c1, tri.c2); b = np.maximum(tri.c1, tri.c2)
    qn = pd.DataFrame(dict(a=a, b=b, q=q2)).groupby(["a", "b"]).q.sum()
    bets = []
    for (x, y), p in qn.items():
        o = qn_odds.get(f"{x}={y}")
        if o and 0 < o <= odds_max and p * o >= ev_min:
            bets.append(dict(combo=f"{x}={y}", stake=stake, odds=float(o), p=round(float(p), 5), ev=round(float(p * o), 3)))
    return sorted(bets, key=lambda d: -d["ev"])
