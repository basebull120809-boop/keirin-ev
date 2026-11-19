"""残差モデルの期待値買い（3連複・2車複）— 紙上運用で確かめる戦略（kind="evr3" / "evr2"）

  確率 : 残差モデル（models/residual.json）の3連単確率から、3連複・2車複の確率を足し上げる
  判定 : 期待値 = 確率 × 判定時オッズ × DRIFT が EV_MIN 以上、かつ判定時オッズ ODDS_MAX 倍以下の目を全部
         （判定時→確定で当たり目のオッズは約7%下がるので DRIFT で割り引いて判定する）
  金額 : 1点 STAKE 円の均等買い

過去検証（2026-01〜11・19,363レース、毎月その前までで学習し直し・確定オッズ・100円均等）:
  3連複＋2車複・期待値1.0以上・50倍以下  12,847レース 平均2.0点 的中14% 回収率107%（95%区間 101〜113%）
  前半(1〜6月) 106% → 後半(7〜11月) 108%
  比較: 展開シナリオ予想で同じ条件 93%、旧モデルでは期待値1.0以上の目がほぼ出ない
確定オッズでの結果なので、判定時オッズで同じになるかをこの紙上記録で確かめる。
"""
from .pools import _key

EV_MIN = 1.0
ODDS_MAX = 50.0
DRIFT = 0.93
STAKE = 100


def _probs(P, cars, combos):
    trio, qn = {}, {}
    for k, (i, j, l) in enumerate(combos):
        p = float(P[k])
        if p <= 0 or not (cars[i] and cars[j] and cars[l]):
            continue
        a, b, c = int(cars[i]), int(cars[j]), int(cars[l])
        kt = _key(f"{a}={b}={c}")
        trio[kt] = trio.get(kt, 0.0) + p
        kq = _key(f"{a}={b}")
        qn[kq] = qn.get(kq, 0.0) + p
    return trio, qn


def _pick(probs, odds):
    out = []
    for k, o in odds.items():
        try:
            o = float(o)
        except (TypeError, ValueError):
            continue
        k = _key(k)
        p = probs.get(k, 0.0)
        if o <= 0 or o > ODDS_MAX or p <= 0:
            continue
        ev = p * o * DRIFT
        if ev >= EV_MIN:
            out.append(dict(combo=k, stake=STAKE, odds=o, p=round(p, 5), ev=round(ev, 3)))
    return sorted(out, key=lambda b: -b["ev"])


def plan(P, cars, combos, trio_odds, qn_odds):
    """P: 残差モデルの3連単確率（504）→ (3連複の買い目, 2車複の買い目)"""
    trio, qn = _probs(P, cars, combos)
    return _pick(trio, trio_odds or {}), _pick(qn, qn_odds or {})
