"""券種の使い分け：3連単モデルの確率から 2車単・3連複 の確率を作り、
レースごとに「どの券種で買うのが一番良いか」を選ぶ。

考え方
  - 2車複 : 強い2人が絞れるレース向き。着順を問わないので2車単より当たりやすい
  - 2車単 : 1着・2着が絞れるレース向き。少ない点数で的中率を取りにいく
  - 3連複 : 上位3人は絞れるが着順が読みにくいレース向き
  - 3連単 : 混戦。フォーメーションで配当を取りにいく（既存の本命予想）
  どの券種も「モデル確率（市場オッズで補正）×その券種の実オッズ」で期待回収を計算し、
  おすすめは「とにかく当てる」：券種ごとに確率の高い順に的中率60%まで広げ、
  合成オッズ2倍（届かなければ最低1.7倍）以上・買い方全体の期待回収 ev_floor（80%）以上を保てる目だけを足していく
  的中率が一番高い券種を選ぶ（期待値では選ばない）。
  3連単は「自信があるとき」だけおすすめにする：確率の高い順に tri_max 点まで
  （合成2倍以上を保って）広げて的中率 tri_conf_hit 以上に届くなら、3連単を最優先。
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class TypeRules:
    budget: int = 3000
    unit: int = 100
    ex_max: int = 8            # 2車単の最大点数
    qn_max: int = 6            # 2車複の最大点数
    ex_synth_min: float = 2.0  # 2車単の合成オッズ下限
    trio_max: int = 15         # 3連複の最大点数
    trio_synth_min: float = 2.5
    model_weight: float = 0.5  # モデルと市場の混ぜ具合（0=市場のみ）
    hit_min: float = 0.20      # おすすめにする的中率の下限
    ev_tie: float = 0.03       # （旧）期待回収の差がこれ以内なら的中率で選ぶ
    hit_target: float = 0.75   # おすすめで狙う的中率（ここまで点数を広げる）
    synth_floor: float = 1.7   # 合成オッズがこれ未満のおすすめは出さない（最低ライン）
    qn_synth_floor: float = 1.7  # 2車複も同じ
    synth_target: float = 2.0  # まず合成オッズこの倍率を保って組む。的中率が足りないときだけ synth_floor まで下げる
    hit_tie: float = 0.03      # 的中率の差がこれ以内なら点数の少ない方
    tri_max: int = 15          # 自信ありの3連単の最大点数
    tri_conf_hit: float = 0.40 # この点数・合成2倍以上でこの的中率に届けば「自信あり」の3連単
    ev_floor: float = 0.85     # おすすめの期待回収の下限（厳選）
    odds_drift: float = 0.93   # 判定時（締切6〜1.5分前）のオッズは、確定時に当たり目ほど下がる（実測で約7%）。損得の計算はこの分を割り引く（買い方全体の 的中率×合成オッズ）


def load_rules(path="config.json"):
    """config.json の "type_rules" で TypeRules を上書き（ev_floor など）"""
    import json
    from pathlib import Path
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8")).get("type_rules", {})
    except Exception:
        cfg = {}
    return TypeRules(**{k: v for k, v in cfg.items() if k in TypeRules.__dataclass_fields__})


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


def hit_plan(pm, odds, kmax, r: TypeRules, blended=False, floor=None):
    """とにかく当てる：確率の高い順に、的中率 hit_target に届くまで広げる。
    足すと 合成オッズが floor を割る目・買い方全体の期待回収（的中率×合成オッズ）が ev_floor を割る目は飛ばす。
    1点目から満たせないなら買わない（None）"""
    p = pm if blended else _blend(pm, odds, r.model_weight)
    if not p:
        return None
    floor = r.synth_floor if floor is None else floor
    sel, acc, inv = [], 0.0, 0.0
    for x in sorted(p, key=p.get, reverse=True):
        if len(sel) >= kmax or acc >= r.hit_target:
            break
        a2, i2 = acc + p[x], inv + 1 / odds[x]
        if 1 / i2 < floor or a2 / i2 < r.ev_floor:
            continue
        sel.append(x); acc, inv = a2, i2
    return _build(sel, p, odds, r, "的中最優先") if sel else None


def _plan(pm, odds, kmax, synth_min, r: TypeRules):
    """おすすめ判定用：的中重視のパターン"""
    ps = [x for x in patterns(pm, odds, kmax, r) if x["name"] == "的中重視"]
    return ps[0] if ps else None


def plans(P, cars, combos, ex_odds, trio_odds, r: TypeRules = TypeRules(), tri_odds=None, qn_odds=None):
    """戻り値: (券種ごとの的中最優先プラン, 1着確率, 券種ごとのパターン一覧)"""
    ex, trio, win, qn = marginals(P, cars, combos)
    d = r.odds_drift
    sc = lambda o: {k: v * d for k, v in o.items()} if o else o     # 確定時の見込みオッズで判定する
    ex_odds, trio_odds, tri_odds, qn_odds = sc(ex_odds), sc(trio_odds), sc(tri_odds), sc(qn_odds)
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
    def two_tier(pm, odds, kmax, floor):
        """合成 synth_target 倍を保って組む。的中率が hit_min に届かないときだけ floor 倍まで下げて組み直す"""
        a = hit_plan(pm, odds, kmax, r, floor=max(floor, r.synth_target))
        if a and a["hit"] >= r.hit_min:
            return a
        b = hit_plan(pm, odds, kmax, r, floor=floor)
        return b if b and (not a or b["hit"] > a["hit"]) else a
    out = {}
    if qn_odds:
        out["qn"] = two_tier(qn, qn_odds, r.qn_max, r.qn_synth_floor)
    if ex_odds:
        out["ex"] = two_tier(ex, ex_odds, r.ex_max, r.synth_floor)
    if trio_odds:
        out["trio"] = two_tier(trio, trio_odds, r.trio_max, r.synth_floor)
    if tri_odds:
        out["tri"] = two_tier(tri, tri_odds, r.tri_max, r.synth_floor)
    out = {k: v for k, v in out.items() if v}
    for v in list(out.values()) + [x for ps in pats.values() for x in ps]:   # 表示する個々のオッズは今のオッズのまま
        for b in v["bets"]:
            b["odds"] = round(b["odds"] / d, 1)
    return out, win, {k: v for k, v in pats.items() if v}


def _summ(bets):
    if not bets:
        return None
    st = np.array([b["stake"] for b in bets], float)
    pp = np.array([b["p"] for b in bets]); oo = np.array([b["odds"] for b in bets])
    return dict(hit=float(pp.sum()), synth=float(1 / (1 / oo).sum()),
                ev=float((st * pp * oo).sum() / st.sum()))


def confident_tri(pl, r: TypeRules = TypeRules()):
    """自信ありの3連単プラン（なければ None）。pl から "tri" を取り除く"""
    t = pl.pop("tri", None)
    if t and t["synth"] >= r.synth_floor and t["hit"] >= r.tri_conf_hit and t["ev"] >= r.ev_floor:
        return t
    return None


def recommend(pl, yoso_bets, win, r: TypeRules = TypeRules(), tri_conf=None):
    """おすすめ券種と理由。pl: plans() の戻り値, yoso_bets: 3連単フォーメーションの買い目"""
    cand = {k: dict(hit=v["hit"], synth=v["synth"], ev=v["ev"]) for k, v in pl.items() if k != "tri"}
    ys = _summ(yoso_bets)
    if ys:                                       # 3連単の買い目は今のオッズのまま → 確定時の見込みに割り引く
        ys["synth"] *= r.odds_drift; ys["ev"] *= r.odds_drift
        cand["yoso"] = ys
    for k, v in cand.items():
        v["n"] = len(pl[k]["bets"]) if k in pl else len(yoso_bets)
    if tri_conf is not None:                     # 自信ありの3連単は最優先
        c = cand["yoso"]
        return "yoso", f"着順まで読める自信あり。3連単{c['n']}点で的中率{c['hit']:.0%}（合成{c['synth']:.1f}倍）", cand
    ok = {k: v for k, v in cand.items()
          if v["synth"] >= (r.qn_synth_floor if k == "qn" else r.synth_floor)
          and v["ev"] >= r.ev_floor and v["hit"] >= r.hit_min}
    if not ok:
        return None, (f"期待回収{r.ev_floor:.0%}以上・的中率{r.hit_min:.0%}以上・合成{r.synth_floor:g}倍以上を満たす券種がないため見送り"), cand
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
        head = f"1着は{a}番が{pa:.0%}で抜けていて、2着まで絞れる" if pa >= 0.45 else f"1着候補の筆頭は{a}番（{pa:.0%}）。上位の1・2着の組み合わせが絞れていて"
        why = f"{head}。2車単なら{len(pl['ex']['bets'])}点で的中率{c['hit']:.0%}"
    elif rec == "trio":
        why = f"上位3人で決まる見込みが高い一方、着順は読みにくい。3連複{len(pl['trio']['bets'])}点で的中率{c['hit']:.0%}"
    else:
        why = f"3連単フォーメーション{c['n']}点が一番当たりやすい（的中率{c['hit']:.0%}）"
    return rec, why, cand


# ---------------- 券種をまたいで1つ選ぶ（2026-11-18 改訂：1点勝負・絞りはしない） ----------------
# 1000回のレースシミュレーション（展開の揺らぎ込み）で、一番多くの回で当たる買い方を選ぶ。
# どの券種も最低点数を設けて、1つの展開・1つの着順に賭ける買い方は出さない。
#   2車複・ワイド 2〜4点／2車単 3〜6点／3連複 4〜10点／3連単 10〜20点
# 合成オッズ1.7倍未満（確定時の見込み）・見込み回収80%未満の買い方は使わない。差が1%以内なら見込み回収の高い方。
# 過去検証（2026-09〜11・2,273R、確定オッズ）：685R 的中39%・回収84%・平均5.4点
#   （参考：前の「見込み回収最大」は 的中27〜30%・回収97% だったが1点勝負が多かった）
MULTI_CANDS = {"qn": [2, 3, 4], "wide": [2, 3, 4], "ex": [3, 4, 5, 6],
               "trio": [4, 5, 6, 8, 10], "tri": [10, 12, 15, 18, 20]}
MULTI_LABEL = {"qn": "2車複", "wide": "ワイド", "ex": "2車単", "trio": "3連複", "tri": "3連単"}
MULTI_EV_MIN = 0.80       # 買い方全体の見込み回収（確定時の見込みオッズで）がこれ未満の買い方は使わない
MULTI_SYNTH_MIN = 1.7     # 合成オッズ（確定時の見込み）がこれ未満の買い方は使わない（2026-11-18 2.0→1.7：2.0では買いが1日数Rしか出ないため）
MULTI_TIE = 0.01          # シミュレーションの的中率の差がこれ以内なら見込み回収の高い方
MULTI_HIT_MIN = 0.30      # シミュレーションの的中率がこれ未満の買い方は使わない（2026-11-18 追加：無理に荒れ目だけを買う予想はせず見送る）


def all_marginals(P, cars, combos):
    """3連単確率 → 券種ごとの確率 {券種: {組: 確率}}。ワイドは「2人とも3着以内」"""
    ex, trio, _, qn = marginals(P, cars, combos)
    tri, wide = {}, {}
    n = len(cars)
    for idx, (i, j, k) in enumerate(combos):
        if max(i, j, k) >= n or P[idx] <= 0:
            continue
        p = float(P[idx]); a, b, c = int(cars[i]), int(cars[j]), int(cars[k])
        tri[f"{a}-{b}-{c}"] = p
        for x, y in ((a, b), (a, c), (b, c)):
            key = f"{min(x, y)}={max(x, y)}"
            wide[key] = wide.get(key, 0) + p
    return dict(qn=qn, wide=wide, ex=ex, trio=trio, tri=tri)


def _sim_keys(finish):
    """1回分の着順 (a,b,c) → 当たりになる組 {券種: set}"""
    a, b, c = finish
    s2 = lambda x, y: f"{min(x, y)}={max(x, y)}"
    return dict(tri={f"{a}-{b}-{c}"}, ex={f"{a}-{b}"}, qn={s2(a, b)},
                trio={"=".join(map(str, sorted((a, b, c))))}, wide={s2(a, b), s2(a, c), s2(b, c)})


def sim_finishes(sim, cars, combos):
    """シミュレーション各回の着順（車番） [(a,b,c), ...]"""
    return [tuple(int(cars[x]) for x in combos[o]) for o in sim["outcome"]]


class _Plans(list):
    """multi_plans の戻り値。blocked = 一番来そうな目が元返しだった券種 [(券種, 組, 判定時オッズ)]"""
    def __init__(self, *a):
        super().__init__(*a)
        self.blocked = []


def multi_plans(P, cars, combos, odds, drift=0.93, cands=None, finishes=None):
    """odds: {券種: {組: 判定時オッズ}}（無い券種は飛ばす）。finishes: シミュレーション各回の着順（あれば的中回数を数える）
    戻り値: 候補の買い方のリスト（確率の高い順に上から k 点。hit=的中確率, sim_hit=シミュレーションで当たった割合,
    synth=合成オッズ, ev=見込み回収。オッズはどれも確定時の見込み）"""
    cands = cands or MULTI_CANDS
    pm = all_marginals(P, cars, combos)
    keys = [_sim_keys(f) for f in finishes] if finishes else None
    out = _Plans()
    for t, ks in cands.items():
        o = {k: float(v) * drift for k, v in (odds.get(t) or {}).items() if v and float(v) > 0}
        p = pm.get(t, {})
        top = max((x for x in p if x in o), key=lambda x: p[x], default=None)
        if top is not None and o[top] <= 1.0:    # 一番来そうな目が元返し（1.0倍前後）→ 本命を外して残りを買うことになるのでレースごと見送り
            out.blocked.append((t, top, round(o[top] / drift, 1)))
        order = sorted((x for x in p if o.get(x, 0) > 1.0), key=lambda x: -p[x])
        for k in ks:
            if len(order) < k:
                break
            sel = order[:k]; ss = set(sel)
            inv = sum(1 / o[x] for x in sel); hit = sum(p[x] for x in sel)
            sh = sum(1 for kk in keys if kk[t] & ss) / len(keys) if keys else hit
            out.append(dict(t=t, k=k, sel=sel, hit=hit, sim_hit=sh, synth=1 / inv, ev=hit / inv,
                            bets=[dict(combo=x, odds=round(o[x] / drift, 1), p=round(p[x], 5),
                                       ev=round(p[x] * o[x], 3)) for x in sel]))
    return out


def multi_pick(plans, ev_min=MULTI_EV_MIN, synth_min=MULTI_SYNTH_MIN, tie=MULTI_TIE, hit_min=MULTI_HIT_MIN):
    """候補から1つ選ぶ：シミュレーションで一番多く当たる買い方。
    戻り値: (選んだ買い方 or None, 理由, 券種ごとの一番当たる候補)"""
    bl = getattr(plans, "blocked", [])
    if bl:                                       # 2026-11-18 追加：本命が元返しのレースは残りの目を買わない
        t, c, o = bl[0]
        return None, f"一番来そうな目（{MULTI_LABEL[t]} {c}）が{o:.1f}倍で、残りの目を買うことになるため見送り", {}
    ok0 = [x for x in plans if x["synth"] >= synth_min and x["ev"] >= ev_min]
    ok = [x for x in ok0 if x["sim_hit"] >= hit_min]
    best_by = {}
    for x in ok:
        if x["t"] not in best_by or x["sim_hit"] > best_by[x["t"]]["sim_hit"]:
            best_by[x["t"]] = x
    summ = {t: dict(n=x["k"], hit=round(x["sim_hit"], 4), synth=round(x["synth"], 2), ev=round(x["ev"], 4))
            for t, x in best_by.items()}
    if ok0 and not ok:
        b0 = max(ok0, key=lambda x: x["sim_hit"])
        return None, (f"条件を満たす買い方は{MULTI_LABEL[b0['t']]}{b0['k']}点（的中{b0['sim_hit']:.0%}）のような"
                      f"荒れ目ねらいだけで、的中{hit_min:.0%}に届かないため見送り"), summ
    if not ok:
        return None, (f"どの券種も、合成{synth_min:g}倍以上・見込み回収{ev_min:.0%}以上で"
                      f"複数点に広げられないため見送り"), summ
    b = max(x["sim_hit"] for x in ok)
    pick = max((x for x in ok if x["sim_hit"] >= b - tie), key=lambda x: x["ev"])
    others = sorted((v for t, v in best_by.items() if t != pick["t"]), key=lambda x: -x["sim_hit"])[:2]
    oth = "・".join(f"{MULTI_LABEL[v['t']]}{v['k']}点 {v['sim_hit']:.0%}" for v in others)
    why = (f"{MULTI_LABEL[pick['t']]}{pick['k']}点：1000回のシミュレーションで{pick['sim_hit']:.0%}的中"
           f"・合成{pick['synth']:.1f}倍・見込み回収{pick['ev']:.0%}"
           + (f"（ほかの券種：{oth}）" if oth else ""))
    return pick, why, summ
