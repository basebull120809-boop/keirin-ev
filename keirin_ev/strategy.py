"""買い目の決め方と資金配分

mode = "points"（標準）: 1レース10〜20点。硬いレースは6点まで絞る
  1) 硬さ    : 3連単1番人気のオッズで判定（hard_odds 倍未満なら硬い → hard_points 点）
               それ以外は 1番人気オッズが hard_odds → wide_odds 倍で min_points → max_points 点
  2) 選び方  : 期待値（モデル確率×オッズ）の高い順。1着候補は最低 min_firsts 人に分散
  3) 配分    : どれが当たっても払戻がほぼ同じになるよう配分（合計 race_budget 円）
  4) 見送り  : 選んだ組の期待値（的中確率×合成オッズ）が race_ev_min 未満なら買わない
mode = "ev": 期待値 ev_min 以上の目だけ、最大 ev_points 点（ケリー配分）
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class Rules:
    mode: str = "points"
    # --- 点数型 ---
    min_points: int = 10
    max_points: int = 20
    hard_points: int = 6
    hard_odds: float = 10.0      # 1番人気がこの倍率未満なら「硬い」
    wide_odds: float = 40.0      # 1番人気がこの倍率以上なら最大点数
    min_firsts: int = 2          # 1着候補の最低人数
    race_ev_min: float = 1.10    # レース全体の期待値の下限（これ未満は見送り）
    race_budget: int = 3000      # 1レースの購入額
    # --- 的中重視型 ---
    synth_min: float = 1.5       # 合成オッズの下限（これ未満は妙味なしで見送り）
    model_weight: float = 0.5    # モデル確率と市場確率の混ぜ具合（データが少ないうちは市場寄り）
    # --- 期待値型 ---
    ev_min: float = 1.20
    ev_points: int = 8
    kelly_frac: float = 0.25
    max_bet_pct: float = 0.01
    max_race_pct: float = 0.03
    # --- 共通 ---
    p_min: float = 0.0           # 点数型では0（広げる前提）
    odds_max: float = 9999.0
    unit: int = 100


def n_points(odds, rules):
    valid = odds[odds > 0]
    if not len(valid):
        return 0
    fav = valid.min()
    if fav < rules.hard_odds:
        return rules.hard_points
    t = min(1.0, (fav - rules.hard_odds) / max(1e-9, rules.wide_odds - rules.hard_odds))
    return int(round(rules.min_points + t * (rules.max_points - rules.min_points)))


def _equal_payout(odds_sel, budget, unit):
    """払戻がそろうように配分（各最低1口）"""
    w = 1.0 / odds_sel
    raw = budget * w / w.sum()
    st = np.maximum(unit, np.floor(raw / unit) * unit)
    return st.astype(int)


def pick_points(p, odds, rules, first_of):
    """戻り値 [(combo_idx, stake, p, odds, ev)]、見送りは []"""
    p = shrink(p, odds, rules.model_weight)
    ev = p * odds
    ok = np.where((odds > 0) & (odds <= rules.odds_max) & (p >= rules.p_min))[0]
    if not len(ok):
        return []
    N = min(n_points(odds, rules), len(ok))
    order = ok[np.argsort(-ev[ok])]
    sel = list(order[:N])
    # 1着候補を最低 min_firsts 人に（足りなければ別の1着の上位目と入れ替え）
    firsts = {first_of[n] for n in sel}
    if len(firsts) < rules.min_firsts:
        for n in order[N:]:
            if first_of[n] not in firsts:
                sel[-1] = n
                firsts = {first_of[m] for m in sel}
                if len(firsts) >= rules.min_firsts:
                    break
    sel = np.array(sel)
    stakes = _equal_payout(odds[sel], rules.race_budget, rules.unit)
    race_ev = float((stakes * p[sel] * odds[sel]).sum() / stakes.sum())
    if race_ev < rules.race_ev_min:
        return []
    keep = np.argsort(-ev[sel])
    return [(int(sel[i]), int(stakes[i]), float(p[sel[i]]), float(odds[sel[i]]), float(ev[sel[i]]))
            for i in keep]


def pick_ev(p, odds, bankroll, rules):
    ev = p * odds
    ok = (odds > 0) & (ev >= rules.ev_min) & (p >= max(rules.p_min, 0.004)) & (odds <= 300)
    idx = np.where(ok)[0]
    idx = idx[np.argsort(-ev[idx])][: rules.ev_points]
    bets = []
    for n in idx:
        o = odds[n]
        f = min(rules.kelly_frac * (p[n] * o - 1) / (o - 1), rules.max_bet_pct)
        bets.append([n, f * bankroll, p[n], o, ev[n]])
    total = sum(b[1] for b in bets)
    scale = min(1.0, rules.max_race_pct * bankroll / total) if total > 0 else 0
    out = []
    for n, st, pp, o, e in bets:
        st = int(st * scale // rules.unit * rules.unit)
        if st >= rules.unit:
            out.append((int(n), st, float(pp), float(o), float(e)))
    return out


def shrink(p, odds, w):
    """モデル確率を市場確率へ寄せる（確率の小さい目の過大評価を抑える）"""
    ok = odds > 0
    q = np.zeros_like(p); q[ok] = 1 / odds[ok]; q /= q.sum()
    out = np.zeros_like(p)
    out[ok] = np.exp(w * np.log(np.maximum(p[ok], 1e-12)) + (1 - w) * np.log(q[ok]))
    return out / out.sum()


def pick_hit(p, odds, rules, first_of):
    """的中重視：当たる確率の高い順にN点。合成オッズ・レース期待値で見送り判定"""
    p = shrink(p, odds, rules.model_weight)
    ok = np.where((odds > 0) & (odds <= rules.odds_max))[0]
    if not len(ok):
        return []
    N = min(n_points(odds, rules), len(ok))
    order = ok[np.argsort(-p[ok])]
    sel = list(order[:N])
    firsts = {first_of[n] for n in sel}
    if len(firsts) < rules.min_firsts:
        for n in order[N:]:
            if first_of[n] not in firsts:
                sel[-1] = n; break
    sel = np.array(sel)
    synth = 1 / (1 / odds[sel]).sum()
    if synth < rules.synth_min:
        return []
    stakes = _equal_payout(odds[sel], rules.race_budget, rules.unit)
    ev = p * odds
    race_ev = float((stakes * p[sel] * odds[sel]).sum() / stakes.sum())
    if race_ev < rules.race_ev_min:
        return []
    keep = np.argsort(-p[sel])
    return [(int(sel[i]), int(stakes[i]), float(p[sel[i]]), float(odds[sel[i]]), float(ev[sel[i]]))
            for i in keep]


def pick_bets(p, odds, bankroll, rules: Rules):
    """p, odds: 1レース分の (504,) 配列"""
    if rules.mode == "hit":
        from .model import COMBOS
        return pick_hit(p, odds, rules, COMBOS[:, 0])
    if rules.mode == "points":
        from .model import COMBOS
        return pick_points(p, odds, rules, COMBOS[:, 0])
    return pick_ev(p, odds, bankroll, rules)


def summary(bets):
    """合成オッズ・的中確率・期待値（レース単位）"""
    if not bets:
        return None
    st = np.array([b[1] for b in bets], float)
    pp = np.array([b[2] for b in bets]); oo = np.array([b[3] for b in bets])
    pay = st * oo                              # 当たったときの払戻（円）
    return dict(total=int(st.sum()), hit_p=float(pp.sum()),
                synth=float(1 / (1 / oo).sum()),
                pay_min=int(pay.min()), pay_max=int(pay.max()),
                race_ev=float((st * pp * oo).sum() / st.sum()))


def formations(combos):
    """['3-6-1','3-6-2','3-1-6',...] → 軸ごとのフォーメーション [(1着, [2着...], [3着...]), ...]
    各行の 1着×2着×3着 の全組み合わせが、選んだ目とぴったり一致するようにまとめる"""
    by_first = {}
    for c in combos:
        a, b, c3 = (int(x) for x in c.split("-"))
        by_first.setdefault(a, {}).setdefault(b, set()).add(c3)
    rows = []
    for a, seconds in by_first.items():
        groups = {}
        for b, thirds in seconds.items():
            groups.setdefault(tuple(sorted(thirds)), []).append(b)
        for thirds, bs in sorted(groups.items(), key=lambda x: (-len(x[0]) * len(x[1]), x[1])):
            rows.append((a, sorted(bs), list(thirds)))
    return rows
