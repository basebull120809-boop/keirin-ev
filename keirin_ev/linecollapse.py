"""ライン崩壊確率 ── ライン単位の確率と、選手の「条件つきのクセ」

崩壊 = 2人以上のラインで、ラインの誰も3着に入らない
市場の見積もり = 3連単オッズから出した「そのラインが3着内に誰もいない」確率
選手のクセ（その日より前のレースだけで計算・レース数で全体平均に縮める）:
  先頭で走ったときの3着内率 − 市場の見積もり（全体／風2m以上の日／333バンク／先行争い3人以上）
  番手で走ったときの 〃（全体／先頭が3着外だったとき＝離れずに残れるか）
レース条件: 風・見なし直線・333・先行争いの数、と先頭が先行型か の掛け合わせ
検証は週ごとのウォークフォワード。市場だけのモデルより良くなるかを見る。
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .upset import Logit, _auc, _ll, calib, SENKO

K = 8                    # クセの縮め具合（このレース数ぶん0に寄せる）
MARKET = ["m_logit"]
COND = ["wind", "z_straight", "short", "n_senko", "head_senko", "size",
        "wind_x_st", "wind_x_senko", "st_x_senko", "wind_x_st_x_senko", "short_x_senko", "wind_x_headsenko"]
TRAIT = ["h_all", "h_wind", "h_short", "h_rival", "s_all", "s_alone"]
FEATS = MARKET + COND + TRAIT
LABEL = dict(m_logit="市場の崩壊見積もり", wind="風速", z_straight="見なし直線", short="333バンク",
             n_senko="先行争いの数", head_senko="先頭が先行型", size="ラインの人数",
             wind_x_st="風×直線", wind_x_senko="風×先行争い", st_x_senko="直線×先行争い",
             wind_x_st_x_senko="風×直線×先行争い", short_x_senko="333×先行争い", wind_x_headsenko="風×先頭が先行型",
             h_all="先頭のクセ（全体）", h_wind="先頭のクセ（風の日）", h_short="先頭のクセ（333）",
             h_rival="先頭のクセ（先行争い多い）", s_all="番手のクセ（全体）", s_alone="番手のクセ（先頭が沈んだとき）")


def _prior(rows, mask, val, base, key="key"):
    """mask の行について、選手ごと・その日より前の (val-base) の合計 /(件数+K) を全行に付ける"""
    r = rows.loc[mask, [key, "date"]].assign(d=(rows.loc[mask, val] - rows.loc[mask, base]).values, n=1.0)
    day = r.groupby([key, "date"]).agg(d=("d", "sum"), n=("n", "sum")).reset_index()
    day["date_"] = pd.to_datetime(day["date"])
    day = day.sort_values("date_")
    day["cd"] = day.groupby(key)["d"].cumsum()
    day["cn"] = day.groupby(key)["n"].cumsum()
    q = rows[[key]].assign(date_=pd.to_datetime(rows["date"]), _i=np.arange(len(rows))).sort_values("date_")
    m = pd.merge_asof(q, day[[key, "date_", "cd", "cn"]], on="date_", by=key,
                      direction="backward", allow_exact_matches=False).sort_values("_i")
    return (m["cd"].fillna(0.0) / (m["cn"].fillna(0.0) + K)).values


def build(con):
    from .perspectives import STRAIGHT, rider_key
    races = pd.read_sql("SELECT race_id, date, venue, bank_len, wind FROM races", con)
    e = pd.read_sql("SELECT race_id, car, name, pref, term, line_id, line_pos, line_size, tactic, style FROM entries", con)
    o = pd.read_sql("SELECT race_id, combo, odds FROM odds_trifecta WHERE snapshot='final' AND odds>0", con)
    res = pd.read_sql("SELECT race_id, first, second, third FROM results", con)
    o["q"] = (1 / o["odds"]) / (1 / o["odds"]).groupby(o["race_id"]).transform("sum")
    cs = o["combo"].str.split("-", expand=True).astype(int)
    m3 = pd.concat([o[["race_id", "q"]].assign(car=cs[i]) for i in range(3)]).groupby(["race_id", "car"])["q"].sum().rename("m3")
    e = e.merge(races, on="race_id").merge(res, on="race_id").join(m3, on=["race_id", "car"])
    e = e.dropna(subset=["m3"])
    e["top3"] = ((e["car"] == e["first"]) | (e["car"] == e["second"]) | (e["car"] == e["third"])).astype(float)
    e["key"] = rider_key(e)
    e["senko"] = e["tactic"].isin(SENKO)
    ns = e.groupby("race_id")["senko"].sum().rename("n_senko")
    e = e.join(ns, on="race_id")
    e["wind"] = e["wind"].fillna(1.0)
    head = (e["line_pos"] == 1) & (e["line_size"] > 1)
    sec = e["line_pos"] == 2
    # 番手のときの先頭の着
    hd = e[head][["race_id", "line_id", "top3"]].rename(columns={"top3": "head_top3"})
    e = e.merge(hd, on=["race_id", "line_id"], how="left")
    e.index = range(len(e))
    head = (e["line_pos"] == 1) & (e["line_size"] > 1)
    sec = e["line_pos"] == 2
    e["h_all"] = _prior(e, head, "top3", "m3")
    e["h_wind"] = _prior(e, head & (e["wind"] >= 2), "top3", "m3")
    e["h_short"] = _prior(e, head & (e["bank_len"] <= 335), "top3", "m3")
    e["h_rival"] = _prior(e, head & (e["n_senko"] >= 3), "top3", "m3")
    e["s_all"] = _prior(e, sec, "top3", "m3")
    e["s_alone"] = _prior(e, sec & (e["head_top3"] == 0), "top3", "m3")

    # ライン単位にまとめる
    L = e[e["line_size"] > 1]
    lines = L.groupby(["race_id", "line_id"]).agg(date=("date", "first"), venue=("venue", "first"),
                                                  bank_len=("bank_len", "first"), wind=("wind", "first"),
                                                  n_senko=("n_senko", "first"), size=("car", "size"),
                                                  any3=("top3", "max"), cars=("car", lambda s: tuple(sorted(s))))
    hh = L[L["line_pos"] == 1].set_index(["race_id", "line_id"])
    ss = L[L["line_pos"] == 2].set_index(["race_id", "line_id"])
    for c in ("h_all", "h_wind", "h_short", "h_rival"):
        lines[c] = hh[c].reindex(lines.index).fillna(0.0)
    lines["head_senko"] = hh["senko"].reindex(lines.index).fillna(False).astype(float)
    for c in ("s_all", "s_alone"):
        lines[c] = ss[c].reindex(lines.index).fillna(0.0)
    # 市場の「ラインの誰も3着内にいない」確率
    oc = o.assign(a=cs[0], b=cs[1], c=cs[2])
    lk = lines.reset_index()[["race_id", "line_id", "cars"]]
    mc = []
    for rid, g in oc.groupby("race_id"):
        ls_ = lk[lk["race_id"] == rid]
        if ls_.empty:
            continue
        A, B, C, Q = g["a"].values, g["b"].values, g["c"].values, g["q"].values
        for lid, cars in zip(ls_["line_id"], ls_["cars"]):
            cc = np.array(cars)
            hit = np.isin(A, cc) | np.isin(B, cc) | np.isin(C, cc)
            mc.append((rid, lid, float(Q[~hit].sum())))
    mc = pd.DataFrame(mc, columns=["race_id", "line_id", "m_coll"]).set_index(["race_id", "line_id"])
    lines = lines.join(mc, how="inner")
    p = lines["m_coll"].clip(1e-4, 1 - 1e-4)
    lines["m_logit"] = np.log(p / (1 - p))
    lines["collapse"] = 1.0 - lines["any3"]
    st = lines["venue"].map(STRAIGHT).fillna(55.0)
    lines["z_straight"] = (st - 55.0) / 7.0
    lines["short"] = (lines["bank_len"].fillna(400) <= 335).astype(float)
    w, sk = lines["wind"] - 1.0, lines["n_senko"] - 2.0
    lines["wind_x_st"] = w * lines["z_straight"]
    lines["wind_x_senko"] = w * sk
    lines["st_x_senko"] = lines["z_straight"] * sk
    lines["wind_x_st_x_senko"] = w * lines["z_straight"] * sk
    lines["short_x_senko"] = lines["short"] * sk
    lines["wind_x_headsenko"] = w * lines["head_senko"]
    for c in FEATS:
        lines[c] = pd.to_numeric(lines[c], errors="coerce").astype(float).fillna(0.0)
    return lines.reset_index().sort_values("date")


def walk_forward(df, min_days=35, step=7, l2=2.0):
    d = pd.to_datetime(df["date"])
    t, out = d.min() + pd.Timedelta(days=min_days), []
    sets = dict(market=MARKET, cond=MARKET + COND, trait=MARKET + TRAIT, full=FEATS)
    while t <= d.max():
        tr, te = df[d < t], df[(d >= t) & (d < t + pd.Timedelta(days=step))]
        if len(te) and len(tr) > 1000:
            p = te[[]].copy()
            for k, f in sets.items():
                p[k] = Logit(f, l2).fit(tr, tr["collapse"]).predict(te)
            out.append(p)
        t += pd.Timedelta(days=step)
    return df.join(pd.concat(out), how="inner")


def report(con, log=print):
    df = build(con)
    wf = walk_forward(df)
    y = wf["collapse"].values
    res = dict(n_lines=int(len(df)), n_test=int(len(wf)), rate=round(float(df["collapse"].mean()), 3),
               market_mean=round(float(df["m_coll"].mean()), 3),
               period=[str(df["date"].min()), str(df["date"].max())])
    res["scores"] = {k: dict(auc=_auc(y, wf[k].values), ll=_ll(y, wf[k].values))
                     for k in ("market", "cond", "trait", "full")}
    res["calib"] = calib(y, wf["full"].values)
    res["over80"] = int((wf["full"] >= .8).sum())
    full = Logit(FEATS, 2.0).fit(df, df["collapse"])
    res["coefs"] = full.coefs()
    # 市場より崩壊しやすいと出たライン：差の大きい順に5段階 → 実際の崩壊率と市場の見積もり
    wf = wf.assign(diff=wf["full"] - wf["m_coll"])
    wf["lv"] = pd.qcut(wf["diff"].rank(method="first"), 5, labels=False) + 1
    res["by_diff"] = [dict(lv=int(k), n=int(len(g)), market=round(float(g["m_coll"].mean()), 3),
                           model=round(float(g["full"].mean()), 3), real=round(float(g["collapse"].mean()), 3))
                      for k, g in wf.groupby("lv")]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/linecollapse.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    Path("reports/linecollapse.md").write_text(_md(res), encoding="utf-8")
    log(_md(res))
    return res


def _md(r):
    nm = dict(market="市場だけ", cond="市場＋レース条件", trait="市場＋選手のクセ", full="全部")
    L = ["# ライン崩壊確率の検証", "",
         f"- 期間 {r['period'][0]}〜{r['period'][1]}・2人以上のライン {r['n_lines']:,}本（週ごとの検証 {r['n_test']:,}本）",
         f"- 実際の崩壊率 {r['rate']:.0%}（市場の見積もり平均 {r['market_mean']:.0%}）", "",
         "| モデル | AUC | 対数損失 |", "|---|---|---|"]
    L += [f"| {nm[k]} | {v['auc']:.4f} | {v['ll']:.4f} |" for k, v in r["scores"].items()]
    L += ["", f"80%以上と出たライン: {r['over80']}", "", "| 予測 | 本数 | 予測平均 | 実際 |", "|---|---|---|---|"]
    L += [f"| {c['band']} | {c['n']} | {c['pred']:.0%} | {c['real']:.0%} |" for c in r["calib"]]
    L += ["", "## モデルと市場の差で5段階（1=市場より崩れにくい、5=崩れやすい）", "",
          "| 段階 | 本数 | 市場 | モデル | 実際 |", "|---|---|---|---|---|"]
    L += [f"| {b['lv']} | {b['n']} | {b['market']:.1%} | {b['model']:.1%} | {b['real']:.1%} |" for b in r["by_diff"]]
    top = sorted(r["coefs"].items(), key=lambda kv: -abs(kv[1]))
    L += ["", "係数（標準化・大きい順）: " + "、".join(f"{LABEL.get(k, k)} {v:+.3f}" for k, v in top)]
    return "\n".join(L) + "\n"
