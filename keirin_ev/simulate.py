"""予想前のレースシミュレーション（モンテカルロ 1000回）

モデルの3連単確率は「平均的な展開」での1本の見積もり。実際のレースは
展開しだいで大きく変わるので、判定のたびにそのレースを1000回走らせて
あらゆる想定を混ぜた確率で買い目を組み、1000回分の結果で採点する。

1回ごとにふるもの:
  1) ラインの展開      : ラインごとに共通の揺らぎ（sigma_line）
                         ＝主導権を取れた／取れずに沈んだ、などの展開
  2) 選手個人の出来    : 選手ごとの揺らぎ（sigma_rider）
  3) 確定オッズ        : 判定時（締切6分前）→確定までのオッズの動き（sigma_odds）
                         払戻（収支）の計算だけに使う

その回の3連単確率 = 元の確率 × exp(1着の揺らぎ + gamma×(2着・3着の揺らぎ))
→ その確率で着順を1つ引く。1000回の平均がシミュレーション後の確率。

揺らぎの大きさは過去データで決めた（学習に使っていない 11/08〜11/14 の506レースで、
実際の着順の当たりやすさが最大になる値）。ラインの揺らぎだけが効き、
個人の揺らぎ・オッズの揺らぎを確率に入れると悪化した。
オッズの動きの大きさは、判定時のオッズと確定オッズの差（113レース）から測った。
"""
import json
from pathlib import Path
import numpy as np
from .model import COMBOS, MAXN

N_SIM = 1000
SIM_PATH = Path("models/sim.json")
DEFAULT = dict(sigma_line=0.4, sigma_rider=0.1, gamma=0.6, sigma_odds=0.25)


def load_params():
    try:
        d = json.loads(SIM_PATH.read_text(encoding="utf-8"))
        return {k: float(d.get(k, v)) for k, v in DEFAULT.items()}
    except Exception:
        return dict(DEFAULT)


def _line_index(T, r):
    line = T.line[r]
    lid = np.where(line >= 0, line, 2000 + np.arange(MAXN))
    return np.unique(lid, return_inverse=True)[1]


def simulate(P, T, r, odds, n=N_SIM, prm=None, seed=None):
    """レース r を n 回走らせる。
    P: 元の3連単確率(504)  odds: 判定時の3連単オッズ(504)
    返り値: P=あらゆる展開を混ぜた確率(504), outcome=各回の着順(n), pay=各回の確定3連単オッズ(n,504)"""
    prm = prm or load_params()
    rng = np.random.default_rng(seed)
    li = _line_index(T, r)
    e = (rng.standard_normal((n, li.max() + 1))[:, li] * prm["sigma_line"]
         + rng.standard_normal((n, MAXN)) * prm["sigma_rider"])
    i, j, k = COMBOS.T
    valid = P > 0
    lp = np.where(valid, np.log(np.where(valid, P, 1)), -np.inf)[None] + e[:, i] + prm["gamma"] * (e[:, j] + e[:, k])
    lp -= lp.max(1, keepdims=True)
    Ps = np.exp(lp)
    Ps /= Ps.sum(1, keepdims=True)
    outcome = (Ps.cumsum(1) < rng.random(n)[:, None]).sum(1).clip(0, len(P) - 1)
    pay = np.where(odds > 0, odds[None] * np.exp(rng.standard_normal((n, len(P))) * prm["sigma_odds"]), 0)
    winners = e.argmax(1)                         # その回で一番伸びた選手（展開の内訳用）
    return dict(P=Ps.mean(0), outcome=outcome, pay=pay, n=n, lines=li, lead=winners)


def _hits(combo, oc):
    if "=" in combo:                               # 3連複 / 2車複
        s = set(int(x) for x in combo.split("="))
        return s == set(oc[:len(s)])
    p = tuple(int(x) for x in combo.split("-"))
    return p == tuple(oc[:len(p)])


def evaluate(sim, cars, bets):
    """買い目（combo・stake・odds・kind）を1000回のレースで採点。
    3連単はその回の確定オッズで払い、他の券種は判定時オッズ×同じ割合の動きで払う"""
    n = sim["n"]
    stake = sum(b["stake"] for b in bets)
    if not stake:
        return None
    sd = load_params()["sigma_odds"]
    move = np.exp(np.random.default_rng(1).standard_normal(n) * sd)    # 3連単以外の確定オッズの動き
    ret = np.zeros(n)
    for t, o in enumerate(sim["outcome"]):
        oc = tuple(int(cars[x]) for x in COMBOS[o])
        for b in bets:
            if _hits(b["combo"], oc):
                od = sim["pay"][t, o] if b.get("kind") in ("yoso", "invest") else b["odds"] * move[t]
                ret[t] += b["stake"] * od
    pl = ret - stake
    hit = ret > 0
    return dict(n=n, hits=int(hit.sum()), plus=int((pl > 0).sum()), torigami=int((hit & (pl <= 0)).sum()),
                roi=round(float(ret.mean() / stake), 4), stake=int(stake),
                p10=int(np.percentile(pl, 10)), p50=int(np.percentile(pl, 50)), p90=int(np.percentile(pl, 90)))


def lead_lines(sim, T, r, cars):
    """1000回のうち、どのラインから1着が出たか（多い順）"""
    li = sim["lines"]
    cnt = {}
    for o in sim["outcome"]:
        w = COMBOS[o][0]
        mem = [int(cars[k]) for k in range(MAXN) if T.mask[r, k] and li[k] == li[w]]
        key = "-".join(map(str, mem))
        cnt[key] = cnt.get(key, 0) + 1
    return sorted(cnt.items(), key=lambda x: -x[1])


def summary_text(ev, lead):
    """ページ・通知用の一言"""
    if not ev:
        return []
    n = ev["n"]
    out = [f"1000回シミュレーション：的中 {ev['hits']}回／プラス収支 {ev['plus']}回（トリガミ {ev['torigami']}回）・平均回収率 {ev['roi']:.0%}"]
    if lead:
        out.append("展開の内訳（1着が出たライン）：" + "、".join(f"{k} {v / n:.0%}" for k, v in lead[:3]))
    return out
