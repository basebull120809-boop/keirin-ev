"""波乱度（荒れる／堅い）── レース単位の確率

目標（結果から作る）:
  are  : 3連単の払戻が 10,000円以上（荒れ）
  kata : 3連単の払戻が 1,500円未満（本命堅調）
材料（すべて締切前にわかるもの）:
  市場  … 3連単の1番人気の確率・1着確率の最大・オッズの散らばり（エントロピー）
  メンバー … 人数・ラインの数・単騎の数・先行型の数・競走得点の散らばりと1位-2位の差・級班
  バンク … 見なし直線・カント・333バンクか・風速
  掛け合わせ … 風×直線、風×先行争い、直線×先行争い、風×直線×先行争い、333×先行争い
学習はロジスティック回帰（L2正則化・scipy）。検証は週ごとのウォークフォワード。
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ARE_YEN, KATA_YEN = 10_000, 1_500
SENKO = ("先行", "押え先", "カマシ")
MARKET = ["fav_q", "fav1", "mkt_ent"]
FEATS = MARKET + ["n_riders", "n_lines", "n_single", "max_line", "n_senko", "n_nige",
                  "score_sd", "score_gap", "is_s", "is_l", "is_g",
                  "z_straight", "z_cant", "short", "wind",
                  "wind_x_st", "wind_x_senko", "st_x_senko", "wind_x_st_x_senko", "short_x_senko"]
LABEL = dict(fav_q="1番人気の確率", fav1="1着本命の確率", mkt_ent="オッズの散らばり",
             n_riders="人数", n_lines="ラインの数", n_single="単騎の数", max_line="最大ラインの人数",
             n_senko="先行争い（先行・押え先・カマシの数）", n_nige="逃げ型の数",
             score_sd="得点の散らばり", score_gap="得点1位と2位の差", is_s="S級", is_l="ガールズ",
             is_g="G3以上", z_straight="見なし直線", z_cant="カント", short="333バンク", wind="風速",
             wind_x_st="風×直線", wind_x_senko="風×先行争い", st_x_senko="直線×先行争い",
             wind_x_st_x_senko="風×直線×先行争い", short_x_senko="333×先行争い")
MODEL = Path("models/upset.json")


def race_features(races: pd.DataFrame, entries: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """レースごとの特徴量（odds は race_id, combo, odds の3連単）"""
    from .perspectives import STRAIGHT, CANT
    o = odds[["race_id", "combo", "odds"]].copy()
    o = o[o["odds"] > 0]
    o["inv"] = 1.0 / o["odds"]
    o["q"] = o["inv"] / o.groupby("race_id")["inv"].transform("sum")
    o["first"] = o["combo"].str.split("-").str[0]
    m = o.groupby("race_id").agg(fav_q=("q", "max"),
                                 mkt_ent=("q", lambda q: float(-(q * np.log(q)).sum())))
    m["fav1"] = o.groupby(["race_id", "first"])["q"].sum().groupby("race_id").max()

    e = entries.copy()
    e["senko"] = e["tactic"].isin(SENKO) if "tactic" in e else False
    e["nige_t"] = e["style"].eq("逃")
    ls = e.groupby(["race_id", "line_id"]).size().rename("ls").reset_index()
    ln = ls.groupby("race_id").agg(n_lines=("ls", lambda s: int((s > 1).sum())),
                                   n_single=("ls", lambda s: int((s == 1).sum())),
                                   max_line=("ls", "max"))

    def gap(s):
        s = s.dropna().sort_values(ascending=False).values
        return float(s[0] - s[1]) if len(s) > 1 else 0.0
    g = e.groupby("race_id").agg(n_senko=("senko", "sum"), n_nige=("nige_t", "sum"),
                                 score_sd=("score", "std"), score_gap=("score", gap))
    r = races.set_index("race_id")[["venue", "grade", "n_riders", "bank_len", "wind"]]
    df = r.join(m, how="inner").join(ln).join(g)
    gr = df["grade"].fillna("")
    df["is_s"] = gr.isin(["SS", "S1", "S2"]).astype(float)
    df["is_l"] = gr.str.startswith("L").astype(float)
    df["is_g"] = 0.0
    st = df["venue"].map(STRAIGHT).fillna(55.0)
    ct = df["venue"].map(CANT).fillna(32.0)
    df["z_straight"] = (st - 55.0) / 7.0
    df["z_cant"] = (ct - 32.0) / 2.5
    df["short"] = (df["bank_len"].fillna(400) <= 335).astype(float)
    df["wind"] = df["wind"].fillna(df["wind"].median() if df["wind"].notna().any() else 1.0)
    w = df["wind"] - 1.0                                 # 平均的な風（約1m/s）からの差
    sk = df["n_senko"].fillna(0) - 2.0                   # 先行型2人を基準にした先行争いの多さ
    df["wind_x_st"] = w * df["z_straight"]
    df["wind_x_senko"] = w * sk
    df["st_x_senko"] = df["z_straight"] * sk
    df["wind_x_st_x_senko"] = w * df["z_straight"] * sk
    df["short_x_senko"] = df["short"] * sk
    for c in FEATS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float).fillna(0.0)
    return df


def load_frame(con):
    races = pd.read_sql("SELECT * FROM races", con)
    entries = pd.read_sql("SELECT * FROM entries", con)
    odds = pd.read_sql("SELECT race_id, combo, odds FROM odds_trifecta WHERE snapshot='final'", con)
    res = pd.read_sql("SELECT race_id, first, second, third, tri_payout FROM results", con)
    df = race_features(races, entries, odds)
    df = df.join(res.set_index("race_id"), how="inner")
    df = df[df["tri_payout"] > 0]
    df["are"] = (df["tri_payout"] >= ARE_YEN).astype(float)
    df["kata"] = (df["tri_payout"] < KATA_YEN).astype(float)
    df["date"] = races.set_index("race_id").loc[df.index, "date"].values
    return df.sort_values("date"), odds


class Logit:
    def __init__(self, feats, l2=1.0):
        self.feats, self.l2 = list(feats), l2

    def fit(self, X: pd.DataFrame, y):
        A = X[self.feats].values
        self.mu, self.sd = A.mean(0), A.std(0) + 1e-9
        Z = np.c_[np.ones(len(A)), (A - self.mu) / self.sd]
        y = np.asarray(y, float)
        n = len(y)

        def f(b):
            u = Z @ b
            ll = np.logaddexp(0, u) - y * u
            pen = 0.5 * self.l2 * (b[1:] ** 2).sum()
            p = 1 / (1 + np.exp(-u))
            return ll.sum() / n + pen / n, (Z.T @ (p - y) + self.l2 * np.r_[0, b[1:]]) / n
        self.b = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
        return self

    def predict(self, X):
        A = (X[self.feats].values - self.mu) / self.sd
        return 1 / (1 + np.exp(-(self.b[0] + A @ self.b[1:])))

    def coefs(self):
        return dict(zip(self.feats, (self.b[1:]).round(4)))

    def to_dict(self):
        return dict(feats=self.feats, l2=self.l2, mu=list(self.mu), sd=list(self.sd), b=list(self.b))

    @classmethod
    def from_dict(cls, d):
        m = cls(d["feats"], d["l2"])
        m.mu, m.sd, m.b = np.array(d["mu"]), np.array(d["sd"]), np.array(d["b"])
        return m


def _auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    r = pd.Series(p).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def walk_forward(df, min_days=28, step_days=7, l2=1.0):
    """週ごとに、それより前のデータだけで学習して次の1週を予測する"""
    d = pd.to_datetime(df["date"])
    start = d.min() + pd.Timedelta(days=min_days)
    out = []
    t = start
    while t <= d.max():
        tr, te = df[d < t], df[(d >= t) & (d < t + pd.Timedelta(days=step_days))]
        if len(te) and len(tr) > 500:
            p = te[[]].copy()
            for y in ("are", "kata"):
                p[f"p_{y}"] = Logit(FEATS, l2).fit(tr, tr[y]).predict(te)
                p[f"b_{y}"] = Logit(MARKET, l2).fit(tr, tr[y]).predict(te)   # 市場だけの比較用
            out.append(p)
        t += pd.Timedelta(days=step_days)
    return df.join(pd.concat(out), how="inner") if out else df.iloc[:0]


def calib(y, p, bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, 1.01)):
    c = pd.cut(p, bins, right=False)
    t = pd.DataFrame({"y": y, "p": p, "c": c}).groupby("c", observed=True).agg(n=("y", "size"), pred=("p", "mean"), real=("y", "mean"))
    return [dict(band=f"{int(i.left*100)}–{min(100, int(i.right*100))}%", n=int(r.n),
                 pred=round(float(r.pred), 3), real=round(float(r.real), 3)) for i, r in t.iterrows()]


def band_roi(wf, odds, key="p_are", q=(0, .2, .4, .6, .8, 1.0)):
    """波乱度の5段階 × オッズ帯ごとに「その帯の3連単を全部100円ずつ買った」ときの回収率"""
    wf = wf.copy()
    wf["lv"] = pd.qcut(wf[key].rank(method="first"), len(q) - 1, labels=False) + 1
    o = odds[odds["race_id"].isin(wf.index)].copy()
    win = wf["first"].astype(int).astype(str) + "-" + wf["second"].astype(int).astype(str) + "-" + wf["third"].astype(int).astype(str)
    o["hit"] = o["combo"].values == win.reindex(o["race_id"]).values
    o["lv"] = wf["lv"].reindex(o["race_id"]).values
    o["band"] = pd.cut(o["odds"], [0, 10, 30, 100, 300, 1e9], labels=["〜10倍", "10〜30倍", "30〜100倍", "100〜300倍", "300倍〜"])
    o["pay"] = np.where(o["hit"], o["odds"], 0.0)
    t = o.groupby(["lv", "band"], observed=True).agg(n=("odds", "size"), pay=("pay", "sum"), hits=("hit", "sum"))
    t["roi"] = t["pay"] / t["n"]
    return t.reset_index()


def report(con, out="reports/upset.md", log=print):
    df, odds = load_frame(con)
    wf = walk_forward(df)
    if wf.empty:
        log("検証できるだけのデータがありません"); return None
    res = dict(n_train=int(len(df)), n_test=int(len(wf)), period=[str(df["date"].min()), str(df["date"].max())],
               are_rate=round(float(df["are"].mean()), 3), kata_rate=round(float(df["kata"].mean()), 3))
    for y in ("are", "kata"):
        res[y] = dict(auc=_auc(wf[y].values, wf[f"p_{y}"].values), auc_market=_auc(wf[y].values, wf[f"b_{y}"].values),
                      ll=_ll(wf[y].values, wf[f"p_{y}"].values), ll_market=_ll(wf[y].values, wf[f"b_{y}"].values),
                      base=_ll(wf[y].values, np.full(len(wf), wf[y].mean())),
                      calib=calib(wf[y].values, wf[f"p_{y}"].values),
                      over80=int((wf[f"p_{y}"] >= .8).sum()))
    full = {y: Logit(FEATS).fit(df, df[y]) for y in ("are", "kata")}
    res["coefs"] = {y: m.coefs() for y, m in full.items()}
    br = band_roi(wf, odds)
    res["band_roi"] = [dict(lv=int(r.lv), band=str(r.band), n=int(r.n), hits=int(r.hits), roi=round(float(r.roi), 3))
                       for r in br.itertuples()]
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    MODEL.write_text(json.dumps({y: m.to_dict() for y, m in full.items()} | dict(trained_at=str(pd.Timestamp.now(tz="Asia/Tokyo"))[:16], n=len(df)),
                                ensure_ascii=False), encoding="utf-8")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(_md(res), encoding="utf-8")
    Path("reports/upset.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(_md(res))
    return res


def _md(r):
    L = [f"# 波乱度（荒れ／堅い）の検証", "",
         f"- 期間 {r['period'][0]}〜{r['period'][1]}・{r['n_train']:,}レース（うち週ごとの検証 {r['n_test']:,}レース）",
         f"- 荒れ（3連単 {ARE_YEN:,}円以上）{r['are_rate']:.0%}／堅い（{KATA_YEN:,}円未満）{r['kata_rate']:.0%}", ""]
    for y, nm in (("are", "荒れ"), ("kata", "堅い")):
        x = r[y]
        L += [f"## {nm}", f"- AUC {x['auc']:.3f}（市場だけ {x['auc_market']:.3f}）",
              f"- 対数損失 {x['ll']:.4f}（市場だけ {x['ll_market']:.4f}／何も見ない {x['base']:.4f}）",
              f"- 80%以上と出たレース: {x['over80']}", "", "| 予測 | レース | 予測平均 | 実際 |", "|---|---|---|---|"]
        L += [f"| {c['band']} | {c['n']} | {c['pred']:.0%} | {c['real']:.0%} |" for c in x["calib"]]
        top = sorted(r["coefs"][y].items(), key=lambda kv: -abs(kv[1]))[:10]
        L += ["", "係数（標準化・大きい順）: " + "、".join(f"{LABEL.get(k, k)} {v:+.2f}" for k, v in top), ""]
    L += ["## 波乱度5段階 × オッズ帯の回収率（帯の3連単を全部買った場合）", "", "| 波乱度 | 帯 | 点数 | 的中 | 回収率 |", "|---|---|---|---|---|"]
    L += [f"| {b['lv']} | {b['band']} | {b['n']:,} | {b['hits']} | {b['roi']:.0%} |" for b in r["band_roi"]]
    return "\n".join(L) + "\n"


def predict_races(races, entries, odds):
    """当日用: race_id → dict(are=, kata=, level=)。モデルがなければ空"""
    if not MODEL.exists():
        return {}
    d = json.loads(MODEL.read_text(encoding="utf-8"))
    df = race_features(races, entries, odds)
    if df.empty:
        return {}
    pa = Logit.from_dict(d["are"]).predict(df)
    pk = Logit.from_dict(d["kata"]).predict(df)
    return {rid: dict(are=round(float(a), 3), kata=round(float(k), 3)) for rid, a, k in zip(df.index, pa, pk)}
