"""期待値判定と資金配分"""
import numpy as np
from dataclasses import dataclass


@dataclass
class Rules:
    ev_min: float = 1.20        # 期待値(確率×オッズ)の下限。1.0ではなく余裕を持たせる
    p_min: float = 0.004        # 確率が低すぎる目は推定誤差が大きいので除外
    odds_max: float = 300.0     # 超高配当は推定が不安定なので上限
    kelly_frac: float = 0.25    # ケリー基準の1/4
    max_bet_pct: float = 0.01   # 1点あたり資金の1%まで
    max_race_pct: float = 0.03  # 1レースあたり資金の3%まで
    max_points: int = 8         # 1レースの最大点数
    unit: int = 100


def pick_bets(p, odds, bankroll, rules: Rules):
    """p, odds: 1レース分の (504,) 配列 → [(combo_idx, stake, p, odds, ev)]"""
    ev = p * odds
    ok = (odds > 0) & (ev >= rules.ev_min) & (p >= rules.p_min) & (odds <= rules.odds_max)
    idx = np.where(ok)[0]
    idx = idx[np.argsort(-ev[idx])][: rules.max_points]
    bets = []
    for n in idx:
        o = odds[n]
        f = rules.kelly_frac * (p[n] * o - 1) / (o - 1)
        f = min(f, rules.max_bet_pct)
        bets.append([n, f * bankroll, p[n], o, ev[n]])
    total = sum(b[1] for b in bets)
    cap = rules.max_race_pct * bankroll
    scale = min(1.0, cap / total) if total > 0 else 0
    out = []
    for n, st, pp, o, e in bets:
        st = int(st * scale // rules.unit * rules.unit)
        if st >= rules.unit:
            out.append((int(n), st, float(pp), float(o), float(e)))
    return out
