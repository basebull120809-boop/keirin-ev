"""券種の使い分け：3連単モデルの確率から 2車単・3連複 の確率を作り、
レースごとに「どの券種で買うのが一番良いか」を選ぶ。

考え方
  - 2車単 : 1着・2着が絞れるレース向き。少ない点数で的中率を取りにいく
  - 3連複 : 上位3人は絞れるが着順が読みにくいレース向き
  - 3連単 : 混戦。フォーメーションで配当を取りにいく（既存の本命予想）
  どの券種も「モデル確率（市場オッズで補正）×その券種の実オッズ」で期待回収を計算し、
  的中率の下限を満たすものの中から期待回収が一番高い券種を「おすすめ」にする。
  差が小さいときは的中率の高い券種を優先（見やすさ・使いやすさ重視）。
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class TypeRules:
    budget: int = 3000
    unit: int = 100
    ex_max: int = 5            # 2車単の最大点数
    ex_synth_min: float = 2.0  # 2車単の合成オッズ下限
    trio_max: int = 6          # 3連複の最大点数
    trio_synth_min: float = 2.5
    model_weight: float = 0.5  # モデルと市場の混ぜ具合（0=市場のみ）
    hit_min: float = 0.20      # おすすめにする的中率の下限
    ev_tie: float = 0.03       # 期待回収の差がこれ以内なら的中率で選ぶ


LABEL = {"ex": "2車単", "trio": "3連複", "yoso": "3連単"}


def marginals(P, cars, combos):
    """P: (504,) 3連単確率, cars: 位置→車番, combos: (504,3) 位置の組
    → ({'a-b': p}, {'a=b=c': p}, {a: 1着確率})"""
    ex, trio, win = {}, {}, {}
    n = len(cars)
    for idx, (i, j, k) in enumerate(combos):
        if i >= n or j >= n or k >= n:
            continue
        p = float(P[idx])
        if p <= 0:
            continue
        a, b, c = int(cars[i]), int(cars[j]), int(cars[k])
        ex[f"{a}-{b}"] = ex.get(f"{a}-{b}", 0) + p
        key = "=".join(map(str, sorted((a, b, c))))
        trio[key] = trio.get(key, 0) + p
        win[a] = win.get(a, 0) + p
    return ex, trio, win


def _blend(pm, odds, w):
    """モデル確率を、その券種のオッズから見た市場確率と幾何平均で混ぜる"""
    keys = [k for k in pm if odds.get(k, 0) > 0]
    if not keys:
        return {}
    q = np.array([1 / odds[k] for k in keys]); q /= q.sum()
    m = np.array([pm[k] for k in keys]); m = m / m.sum()
    u = np.exp(w * np.log(np.maximum(m, 1e-12)) + (1 - w) * np.log(q))
    u /= u.sum()
    return dict(zip(keys, u))


def _stakes(odds_sel, budget, unit):
    w = 1.0 / np.asarray(odds_sel)
    raw = budget * w / w.sum()
    return np.maximum(unit, np.floor(raw / unit) * unit).astype(int)


def _plan(pm, odds, kmax, synth_min, r: TypeRules):
    """確率の高い順に最大 kmax 点。合成オッズが下限を割る手前で止める"""
    p = _blend(pm, odds, r.model_weight)
    if not p:
        return None
    order = sorted(p, key=p.get, reverse=True)
    sel = []
    for k in order[:kmax * 3]:                       # 確率の高い順に、合成オッズの下限を割らない目だけ足す
        if len(sel) >= kmax:
            break
        trial = sel + [k]
        if 1 / sum(1 / odds[x] for x in trial) >= synth_min:
            sel = trial
    if not sel:
        return None                                  # 妙味のある目がない
    o = np.array([odds[x] for x in sel]); pp = np.array([p[x] for x in sel])
    st = _stakes(o, r.budget, r.unit)
    return dict(bets=[dict(combo=x, stake=int(s), odds=float(oo), p=round(float(q), 5), ev=round(float(q * oo), 3))
                      for x, s, oo, q in zip(sel, st, o, pp)],
                hit=float(pp.sum()), synth=float(1 / (1 / o).sum()),
                ev=float((st * pp * o).sum() / st.sum()))


def plans(P, cars, combos, ex_odds, trio_odds, r: TypeRules = TypeRules()):
    ex, trio, win = marginals(P, cars, combos)
    out = {}
    if ex_odds:
        out["ex"] = _plan(ex, ex_odds, r.ex_max, r.ex_synth_min, r)
    if trio_odds:
        out["trio"] = _plan(trio, trio_odds, r.trio_max, r.trio_synth_min, r)
    return {k: v for k, v in out.items() if v}, win


def _summ(bets):
    if not bets:
        return None
    st = np.array([b["stake"] for b in bets], float)
    pp = np.array([b["p"] for b in bets]); oo = np.array([b["odds"] for b in bets])
    return dict(hit=float(pp.sum()), synth=float(1 / (1 / oo).sum()),
                ev=float((st * pp * oo).sum() / st.sum()))


def recommend(pl, yoso_bets, win, r: TypeRules = TypeRules()):
    """おすすめ券種と理由。pl: plans() の戻り値, yoso_bets: 3連単フォーメーションの買い目"""
    cand = {k: dict(hit=v["hit"], synth=v["synth"], ev=v["ev"]) for k, v in pl.items()}
    ys = _summ(yoso_bets)
    if ys and ys["synth"] >= 1.5:
        cand["yoso"] = ys
    ok = {k: v for k, v in cand.items() if v["hit"] >= r.hit_min}
    if not ok:
        return None, "どの券種も的中率と配当の釣り合いが取れないため、無理に買わないレースです", cand
    best_ev = max(v["ev"] for v in ok.values())
    near = {k: v for k, v in ok.items() if v["ev"] >= best_ev - r.ev_tie}
    rec = max(near, key=lambda k: near[k]["hit"])
    top = sorted(win.items(), key=lambda x: -x[1])
    a, pa = top[0] if top else (None, 0)
    c = cand[rec]
    if rec == "ex":
        why = f"1着は{a}番が{pa:.0%}で抜けていて、2着まで絞れる。2車単なら{len(pl['ex']['bets'])}点で的中率{c['hit']:.0%}"
    elif rec == "trio":
        why = f"上位3人で決まる見込みが高い一方、着順は読みにくい。3連複{len(pl['trio']['bets'])}点で的中率{c['hit']:.0%}"
    else:
        why = f"混戦（1着候補の最上位でも{pa:.0%}）。3連単のフォーメーションで配当を取りにいく"
    return rec, why, cand
