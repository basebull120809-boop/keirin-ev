"""券種の使い分け：3連単モデルの確率から 2車単・3連複 の確率を作り、
レースごとに「どの券種で買うのが一番良いか」を選ぶ。

考え方
  - 2車複 : 強い2人が絞れるレース向き。着順を問わないので2車単より当たりやすい
  - 2車単 : 1着・2着が絞れるレース向き。少ない点数で的中率を取りにいく
  - 3連複 : 上位3人は絞れるが着順が読みにくいレース向き
  - 3連単 : 混戦。フォーメーションで配当を取りにいく（既存の本命予想）
  どの券種も「モデル確率（市場オッズで補正）×その券種の実オッズ」で期待回収を計算し、
  おすすめは「とにかく当てる」：券種ごとに確率の高い順に的中率60%まで広げ、
  的中率が一番高い券種を選ぶ（期待値では選ばない）。
  ただし当たっても損になる組み方（合成オッズ synth_floor 未満）は避ける。
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class TypeRules:
    budget: int = 3000
    unit: int = 100
    ex_max: int = 5            # 2車単の最大点数
    qn_max: int = 4            # 2車複の最大点数
    ex_synth_min: float = 2.0  # 2車単の合成オッズ下限
    trio_max: int = 10         # 3連複の最大点数
    trio_synth_min: float = 2.5
    model_weight: float = 0.5  # モデルと市場の混ぜ具合（0=市場のみ）
    hit_min: float = 0.20      # おすすめにする的中率の下限
    ev_tie: float = 0.03       # （旧）期待回収の差がこれ以内なら的中率で選ぶ
    hit_target: float = 0.60   # おすすめで狙う的中率（ここまで点数を広げる）
    synth_floor: float = 1.2   # 当たったときに投資の1.2倍は戻る組み方まで（トリガミ防止）
    hit_tie: float = 0.03      # 的中率の差がこれ以内なら点数の少ない方


LABEL = {"qn": "2車複", "ex": "2車単", "trio": "3連複", "yoso": "3連単"}


def marginals(P, cars, combos):
    """P: (504,) 3連単確率, cars: 位置→車番, combos: (504,3) 位置の組
    → ({'a-b': p}, {'a=b=c': p}, {a: 1着確率}, {'a=b': p})"""
    ex, trio, win, qn = {}, {}, {}, {}
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
        q = "=".join(map(str, sorted((a, b))))
        qn[q] = qn.get(q, 0) + p
    return ex, trio, win, qn


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


def _build(sel, p, odds, r: TypeRules, name=""):
    o = np.array([odds[x] for x in sel]); pp = np.array([p[x] for x in sel])
    st = _stakes(o, r.budget, r.unit)
    return dict(name=name,
                bets=[dict(combo=x, stake=int(s), odds=float(oo), p=round(float(q), 5), ev=round(float(q * oo), 3))
                      for x, s, oo, q in zip(sel, st, o, pp)],
                hit=float(pp.sum()), synth=float(1 / (1 / o).sum()),
                ev=float((st * pp * o).sum() / st.sum()), low=bool(1 / (1 / o).sum() < 2.0))


def _prefix(order, odds, kmax, synth_min):
    """確率の高い順に上から何点まで買えるか（合成オッズが synth_min 以上を保てる最大の点数）"""
    k, inv = 0, 0.0
    for i, x in enumerate(order[:kmax]):
        inv += 1 / odds[x]
        if 1 / inv >= synth_min:
            k = i + 1
        else:
            break
    return k


PATTERNS = [("的中重視", 2.0), ("バランス", 4.0), ("高配当", 8.0)]


def patterns(pm, odds, kmax, r: TypeRules, blended=False):
    """予想（確率）の順番はそのままに、点数を変えた買い方を的中率の高い順に並べる。
      的中重視 : 合成オッズ2倍を保てるところまで広げる（本命が安すぎるときは1点・低配当）
      バランス : 合成オッズ4倍を保てるところまで
      高配当   : 合成オッズ8倍を保てるところまで
      穴狙い   : 30倍以上の目だけを、確率の高い順に
      手堅く   : 配当は気にせず、的中率60%に届くまで広げる（先頭に置く。おすすめ判定には使わない）
    """
    p = pm if blended else _blend(pm, odds, r.model_weight)
    if not p:
        return []
    order = sorted(p, key=p.get, reverse=True)
    out, seen = [], set()
    # 手堅く：配当は気にせず、的中率60%に届くまで確率の高い順に広げる（点数の上限まで）
    acc, k60 = 0.0, 0
    for i, x in enumerate(order[:kmax]):
        acc += p[x]; k60 = i + 1
        if acc >= 0.60:
            break
    wide = tuple(order[:k60])
    for name, sm in PATTERNS:
        k = _prefix(order, odds, kmax, sm)
        if k == 0:
            if name != "的中重視":
                continue
            k = 1                                       # 予想が安い本命なら、それ1点
        key = tuple(order[:k])
        if key in seen:
            continue
        seen.add(key); out.append(_build(list(key), p, odds, r, name))
    if wide and wide not in seen:
        out.insert(0, _build(list(wide), p, odds, r, "手堅く")); seen.add(wide)
    ana = [x for x in order if odds[x] >= 30][: max(2, kmax // 3)]
    if ana and tuple(ana) not in seen:
        out.append(_build(ana, p, odds, r, "穴狙い"))
    return out


def hit_plan(pm, odds, kmax, r: TypeRules, blended=False):
    """とにかく当てる：確率の高い順に、的中率 hit_target に届くまで広げる。
    合成オッズが synth_floor を割る（当たっても損に近い）手前で止める。最低1点"""
    p = pm if blended else _blend(pm, odds, r.model_weight)
    if not p:
        return None
    order = sorted(p, key=p.get, reverse=True)
    k, acc, inv = 0, 0.0, 0.0
    for i, x in enumerate(order[:kmax]):
        if i and 1 / (inv + 1 / odds[x]) < r.synth_floor:
            break
        inv += 1 / odds[x]; acc += p[x]; k = i + 1
        if acc >= r.hit_target:
            break
    return _build(order[:max(1, k)], p, odds, r, "的中最優先")


def _plan(pm, odds, kmax, synth_min, r: TypeRules):
    """おすすめ判定用：的中重視のパターン"""
    ps = [x for x in patterns(pm, odds, kmax, r) if x["name"] == "的中重視"]
    return ps[0] if ps else None


def plans(P, cars, combos, ex_odds, trio_odds, r: TypeRules = TypeRules(), tri_odds=None, qn_odds=None):
    """戻り値: (券種ごとの的中最優先プラン, 1着確率, 券種ごとのパターン一覧)"""
    ex, trio, win, qn = marginals(P, cars, combos)
    pats = {}
    if qn_odds:
        pats["qn"] = patterns(qn, qn_odds, r.qn_max, r)
    if ex_odds:
        pats["ex"] = patterns(ex, ex_odds, r.ex_max, r)
    if trio_odds:
        pats["trio"] = patterns(trio, trio_odds, r.trio_max, r)
    if tri_odds:
        tri = {}
        for idx, (i, j, k) in enumerate(combos):
            if max(i, j, k) < len(cars) and P[idx] > 0:
                tri[f"{int(cars[i])}-{int(cars[j])}-{int(cars[k])}"] = float(P[idx])
        pats["tri"] = patterns(tri, tri_odds, 20, r)
    out = {}
    if qn_odds:
        out["qn"] = hit_plan(qn, qn_odds, r.qn_max, r)
    if ex_odds:
        out["ex"] = hit_plan(ex, ex_odds, r.ex_max, r)
    if trio_odds:
        out["trio"] = hit_plan(trio, trio_odds, r.trio_max, r)
    out = {k: v for k, v in out.items() if v}
    return out, win, {k: v for k, v in pats.items() if v}


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
    if ys:
        cand["yoso"] = ys
    for k, v in cand.items():
        v["n"] = len(pl[k]["bets"]) if k in pl else len(yoso_bets)
    ok = {k: v for k, v in cand.items() if v["synth"] >= r.synth_floor}
    if not ok:
        return None, "どの券種も当たったときの払戻が投資を下回る（トリガミ）ため見送り", cand
    best = max(v["hit"] for v in ok.values())
    near = {k: v for k, v in ok.items() if v["hit"] >= best - r.hit_tie}
    rec = min(near, key=lambda k: (near[k]["n"], -near[k]["hit"]))
    top = sorted(win.items(), key=lambda x: -x[1])
    a, pa = top[0] if top else (None, 0)
    c = cand[rec]
    if rec in pl and pl[rec].get("low"):
        return rec, f"{LABEL[rec]}{c['n']}点で的中率{c['hit']:.0%}。本命なので配当は安め（合成{c['synth']:.1f}倍）", cand
    if rec == "qn":
        why = f"強い2人で決まる見込みが高い。着順を問わない2車複{c['n']}点で的中率{c['hit']:.0%}"
    elif rec == "ex":
        why = f"1着は{a}番が{pa:.0%}で抜けていて、2着まで絞れる。2車単なら{len(pl['ex']['bets'])}点で的中率{c['hit']:.0%}"
    elif rec == "trio":
        why = f"上位3人で決まる見込みが高い一方、着順は読みにくい。3連複{len(pl['trio']['bets'])}点で的中率{c['hit']:.0%}"
    else:
        why = f"3連単フォーメーション{c['n']}点が一番当たりやすい（的中率{c['hit']:.0%}）"
    return rec, why, cand
