"""展開予想：どのラインがレースを動かす（主導権を取る）か

段1  ライン単位の条件付きロジット。各ラインの先頭（単騎は本人）の
     B回数・S回数・逃げ／捲り回数・逃げの割合・競走得点（レース内で標準化）・記者の展開予想（先行／押え先／カマシ…）・
     記者の印・ライン人数・単騎 から「そのラインが主導権を取る確率」を出す。
     正解は今は B（最終バックを先頭で通過）を取った選手のライン。勝敗因から H（先行）が取れたら H に切り替える（target）。
使い方
     ・当日の判定で、主導権確率が LEAD_MIN 以上のラインがあれば「先頭→番手」の2車単1点を紙上で記録（kind="tenkai"）
       過去データでは、主導権を取りそうなラインの「先頭→番手」は「番手→先頭」より明らかに安く放置されていた
       （2026/04〜11 確定オッズ：回収率 85% 対 71%）。100%には届いていないので、紙上で実績を見るための枠
     ・一言解説に「主導権予想」を出す
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

LF = ["z_back_cnt", "z_s_cnt", "z_nige", "z_makuri", "z_score", "nige_share", "top_back",
      "t_先行", "t_押え先", "t_カマシ", "t_自在", "t_まくり", "t_追", "tip_v", "lsize", "is_single"]
TIP = {"◎": 3, "○": 2, "▲": 1.5, "△": 1, "×": 0.5, "注": 0.5}
LEAD_MIN = 0.5          # 2車単を記録する主導権確率の下限
STAKE = 1000
PATH = "models/tenkai.json"


def lines(ent):
    """出走表（1レース以上）→ ライン単位の表（先頭の選手・番手の選手・特徴量）"""
    e = ent.copy()
    e["race_id"] = e["race_id"].astype(str)
    e["single"] = e["line_size"].fillna(1) <= 1
    e["lkey"] = np.where(e["single"], "s" + e["car"].astype(str), "l" + e["line_id"].astype(str))
    h = e[(e["line_pos"] == 1) | e["single"]].copy()
    sec = e[(e["line_pos"] == 2) & ~e["single"]][["race_id", "lkey", "car"]].rename(columns={"car": "second_car"})
    h = h.merge(sec, on=["race_id", "lkey"], how="left")
    g = h.groupby("race_id")
    for c in ["back_cnt", "s_cnt", "nige", "makuri", "sashi", "mark", "score"]:
        h[c] = pd.to_numeric(h.get(c), errors="coerce").fillna(0) if c in h else 0.0
    for c in ["back_cnt", "s_cnt", "nige", "makuri", "score"]:
        sd = g[c].transform(lambda s: s.std(ddof=0)).replace(0, 1)
        h["z_" + c] = (h[c] - g[c].transform("mean")) / sd
    h["top_back"] = (g["back_cnt"].rank(ascending=False, method="min") == 1).astype(float)
    h["nige_share"] = h["nige"] / (h["nige"] + h["makuri"] + h["sashi"] + h["mark"] + 1)
    t = h["tactic"].fillna("").astype(str) if "tactic" in h else pd.Series("", index=h.index)
    for k in ["先行", "押え先", "カマシ", "自在", "まくり", "追"]:
        h["t_" + k] = t.str.contains(k).astype(float)
    h["tip_v"] = h["tip"].map(TIP).fillna(0) if "tip" in h else 0.0
    h["lsize"] = h["line_size"].fillna(1)
    h["is_single"] = h["single"].astype(float)
    for c in LF:
        h[c] = h[c].astype(float)
    return h.sort_values(["race_id", "lkey"]).reset_index(drop=True), e


def _fit(H, y, l2=0.01):
    X = H[LF].values
    _, gi = np.unique(H["race_id"], return_inverse=True)
    G = gi.max() + 1

    def nll(b):
        u = X @ b
        m = np.full(G, -1e9); np.maximum.at(m, gi, u)
        s = np.zeros(G); np.add.at(s, gi, np.exp(u - m[gi]))
        return -(u - m[gi] - np.log(s[gi]))[y].sum() / G + l2 * (b ** 2).sum()
    return minimize(nll, np.zeros(len(LF)), method="L-BFGS-B").x


def predict(H, beta):
    u = H[LF].values @ np.asarray(beta)
    rid = H["race_id"].values
    u = u - pd.Series(u).groupby(rid).transform("max").values
    z = np.exp(u)
    return z / pd.Series(z).groupby(rid).transform("sum").values


def _labels(target):
    """(race_id, car) → その選手が主導権を取ったか"""
    from . import store
    F = []
    for d in store.collected_dates():
        f = store.read(d, "finish")
        if f.empty:
            continue
        if target == "H":
            if "reason" not in f or not f["reason"].notna().any() or (f["reason"] == "-").all():
                continue
            from .hrider import mark_h
            f = mark_h(f.assign(kimarite=f.get("kimarite")))
            f["y"] = f["H"]
        else:
            f["y"] = f["sb"].fillna("").astype(str).str.contains("B")
        F.append(f[["race_id", "car", "y"]])
    return pd.concat(F, ignore_index=True) if F else pd.DataFrame(columns=["race_id", "car", "y"])


def train(target="B", path=PATH, log=print):
    from . import store
    lab = _labels(target)
    days = sorted({str(r)[2:10] for r in lab["race_id"]})
    E = [store.read(d, "entries") for d in store.collected_dates()]
    ent = pd.concat([x for x in E if not x.empty], ignore_index=True)
    ent = ent[ent["race_id"].astype(str).isin(set(lab["race_id"].astype(str)))]
    H, e = lines(ent)
    e = e.merge(lab.assign(race_id=lab["race_id"].astype(str)), on=["race_id", "car"], how="left")
    ly = e.groupby(["race_id", "lkey"])["y"].any().rename("y").reset_index()
    H = H.merge(ly, on=["race_id", "lkey"], how="left")
    H["y"] = H["y"].fillna(False).astype(bool)
    ok = H.groupby("race_id")["y"].sum() == 1
    H = H[H["race_id"].isin(ok[ok].index)].reset_index(drop=True)
    # 学習に使っていない期間（最後の2割のレース）での的中率も記録
    rids = sorted(H["race_id"].unique(), key=lambda r: r[2:10])
    cut = rids[int(len(rids) * 0.8)]
    tr, te = H[H["race_id"].map(lambda r: r[2:10]) < cut[2:10]], H[H["race_id"].map(lambda r: r[2:10]) >= cut[2:10]]
    acc = None
    if len(tr) and len(te):
        b = _fit(tr, tr["y"].values)
        te = te.assign(p=predict(te, b))
        top = te.loc[te.groupby("race_id")["p"].idxmax()]
        base = te.loc[te.groupby("race_id")["back_cnt"].idxmax()]
        acc = dict(model=round(float(top["y"].mean()), 4), back_max=round(float(base["y"].mean()), 4),
                   races=int(te["race_id"].nunique()))
    beta = _fit(H, H["y"].values)
    d = dict(target=target, features=LF, beta=list(map(float, beta)), n_races=int(H["race_id"].nunique()),
             oos=acc, trained_at=f"{pd.Timestamp.now(tz='Asia/Tokyo'):%Y-%m-%d %H:%M}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"展開予想（{target}）学習 {d['n_races']}レース  検証期間の的中率 {acc}")
    return d


def load(path=PATH):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d["features"] != LF:
        raise ValueError("展開予想の特徴量が変わっています。python main.py tenkai で学習し直してください")
    return d


def race_plan(ent, exo, names=None, model=None):
    """1レース分：主導権予想と、記録する2車単（先頭→番手）。exo は {"a-b": オッズ}"""
    model = model or load()
    H, _ = lines(ent)
    if H.empty:
        return None, []
    H = H.assign(p=predict(H, model["beta"])).sort_values("p", ascending=False)
    top = H.iloc[0]
    names = names or {}
    who = f"{int(top['car'])}{names.get(int(top['car']), '')}"
    note = f"主導権予想：{who}のライン {top['p']:.0%}" + (
        f"（2番手は {int(H.iloc[1]['car'])}{names.get(int(H.iloc[1]['car']), '')} {H.iloc[1]['p']:.0%}）" if len(H) > 1 else "")
    bets = []
    if top["p"] >= LEAD_MIN and top["second_car"] == top["second_car"]:
        combo = f"{int(top['car'])}-{int(top['second_car'])}"
        o = exo.get(combo)
        if o:
            o = float(o)
            bets.append(dict(combo=combo, stake=STAKE, odds=o, p=round(float(top["p"]), 4), ev=0.0))   # p は主導権確率
            note += f" → 逃げ切り目 2車単 {combo}（{o:.1f}倍）"
    return note, bets
