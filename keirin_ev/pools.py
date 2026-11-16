"""券種どうしのオッズのズレ（3連複）

3連単のオッズから作った「3連複でこの3人」の確率 と、3連複のオッズから読める確率を比べる。
  割安度 r = 3連単から見た確率 ÷ 3連複オッズの確率
r が大きい（3連単は「もっと来る」と見ている）のに3連複が安く売られている目を買う。
3連単はお金が一番集まるので見立てが正確、という考え方。

過去データ（確定オッズ、ガールズ除く）：r≥1.2・20倍以下の3連複を100円ずつ
  4〜8月 回収率 94.4%、9〜11月 100.8%（95%区間 89〜114%）。何も考えずに買うと約75%。
確定オッズでの結果なので、判定時（締切6分前）のオッズでも同じか、紙上で記録して確かめる（kind="pool"）。
"""
import numpy as np

R_MIN = 1.2        # 割安度の下限
ODDS_MAX = 20.0    # 3連複オッズの上限
MAX_POINTS = 3
STAKE = 500


def _key(c):
    return "=".join(map(str, sorted(int(x) for x in str(c).replace("-", "=").split("="))))


def trio_plan(tri_odds, trio_odds):
    """tri_odds: {"a-b-c": オッズ}（3連単）、trio_odds: {"a=b=c": オッズ}（3連複）→ [買い目dict]"""
    tri = {k: float(v) for k, v in tri_odds.items() if v and float(v) > 0}
    tro = {_key(k): float(v) for k, v in trio_odds.items() if v and float(v) > 0}
    if not tri or not tro:
        return []
    zt = sum(1 / v for v in tri.values())
    q3 = {}
    for k, v in tri.items():
        kk = _key(k)
        q3[kk] = q3.get(kk, 0.0) + (1 / v) / zt
    zo = sum(1 / v for v in tro.values())
    cand = []
    for k, o in tro.items():
        q = (1 / o) / zo
        r = q3.get(k, 0.0) / q if q > 0 else 0
        if r >= R_MIN and o <= ODDS_MAX:
            cand.append((r, k, o, q3.get(k, 0.0)))
    cand.sort(reverse=True)
    return [dict(combo=k, stake=STAKE, odds=o, p=round(p, 5), ev=round(p * o, 3), ratio=round(r, 3))
            for r, k, o, p in cand[:MAX_POINTS]]


# ---------------- 当てる予想（3連複4点） ----------------
HIT_POINTS = 4
HIT_P_MIN = 0.5        # 4点の合計の当たる見込み（モデル）
HIT_SYNTH_MIN = 1.5    # 確定時の合成オッズの下限（判定時オッズ × DRIFT で見通す）
DRIFT = 0.93           # 判定時→確定で当たり目のオッズは約7%下がる（実測）
HIT_BUDGET = 2000


def hit_plan(P, cars, combos, trio_odds):
    """モデルの3連単確率 P（504）から3連複の上位4点。見込み50%以上・見通した合成オッズ1.5倍以上のときだけ。
    払戻がそろうように配分（どれが当たっても約 合成オッズ 倍）。過去（10〜11月 確定オッズ）：的中率58%・回収率92%"""
    from .strategy import _equal_payout
    tro = {_key(k): float(v) for k, v in trio_odds.items() if v and float(v) > 0}
    pt = {}
    for n, (i, j, k) in enumerate(combos):
        if P[n] > 0 and cars[i] and cars[j] and cars[k]:
            kk = _key(f"{cars[i]}={cars[j]}={cars[k]}")
            pt[kk] = pt.get(kk, 0.0) + float(P[n])
    top = [k for k in sorted(pt, key=pt.get, reverse=True) if k in tro][:HIT_POINTS]
    if len(top) < HIT_POINTS:
        return [], None
    p = sum(pt[k] for k in top)
    o = np.array([tro[k] for k in top])
    synth = 1 / (1 / (o * DRIFT)).sum()
    info = dict(p=round(p, 3), synth=round(float(synth), 2))
    if p < HIT_P_MIN or synth < HIT_SYNTH_MIN:
        return [], info
    st = _equal_payout(o, HIT_BUDGET, 100)
    if (st * o * DRIFT).min() < st.sum():          # 当たっても損になる目があれば出さない
        return [], info
    return [dict(combo=k, stake=int(s), odds=float(oo), p=round(pt[k], 5), ev=round(pt[k] * oo, 3), synth=info["synth"])
            for k, s, oo in zip(top, st, o)], info
