"""検証用の合成データ（動作確認・システムの検出力テスト用）

bias=0   : 市場が完全に正しい → どんな戦略でも回収率は約75%に収束するはず
bias>0   : 市場が人気側を買いすぎ・ライン連動を過小評価 → 戦略が歪みを拾えるかを確認
"""
from itertools import permutations
import numpy as np
import pandas as pd

VENUES = [("前橋", 335), ("松戸", 333), ("いわき平", 400), ("弥彦", 400),
          ("宇都宮", 500), ("小倉", 400), ("岸和田", 400), ("取手", 500)]


def _lines(n, rng):
    sizes = []
    left = n
    while left > 0:
        s = int(rng.choice([1, 2, 3, 3, 4] if left >= 4 else range(1, left + 1)))
        s = min(s, left); sizes.append(s); left -= s
    return sizes


def generate(n_races=6000, start="2024-01-01", bias=0.25, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_races // 16 + 1, freq="D")
    R, E, RES, OD = [], [], [], []
    for r in range(n_races):
        d = dates[r // 16].strftime("%Y-%m-%d")
        v, bank = VENUES[rng.integers(len(VENUES))]
        n = int(rng.choice([7, 9, 9, 9]))
        rid = f"{d.replace('-', '')}_{r % 16:02d}"
        wind = round(float(rng.gamma(2, 1.5)), 1)
        R.append(dict(race_id=rid, date=d, venue=v, race_no=r % 12 + 1, grade="F1",
                      session="デイ", n_riders=n, bank_len=bank, straight=None,
                      wind=wind, note=None))
        sizes = _lines(n, rng)
        ability = rng.normal(0, 1, n)
        cars = rng.permutation(np.arange(1, n + 1))
        k = 0; line_of = np.zeros(n, int); pos = np.zeros(n, int); lsz = np.zeros(n, int)
        for li, s in enumerate(sizes):
            for p in range(s):
                line_of[k] = li; pos[k] = p + 1; lsz[k] = s; k += 1
        head = (pos == 1) & (lsz > 1)
        true_s = (1.2 * ability + 0.35 * (pos == 2) + 0.15 * head
                  - 0.4 * (lsz == 1) + 0.25 * head * (bank <= 335)
                  - 0.05 * head * wind)
        for i in range(n):
            E.append(dict(race_id=rid, car=int(cars[i]), name=f"選手{r}_{i}",
                          pref=None, **{"class": "A1"},
                          style="逃" if pos[i] == 1 else "追",
                          score=round(90 + 5 * ability[i] + rng.normal(0, 1.5), 2),
                          win_rate=round(max(0, 15 + 10 * ability[i] + rng.normal(0, 5)), 1),
                          ren_rate=round(max(0, 35 + 12 * ability[i] + rng.normal(0, 6)), 1),
                          san_rate=None, back_cnt=None, line_id=int(line_of[i]),
                          line_pos=int(pos[i]), line_size=int(lsz[i])))
        same = line_of[:, None] == line_of[None, :]
        np.fill_diagonal(same, False)

        keys = list(permutations(range(n), 3))
        I, J, K = np.array(keys).T
        eye = np.eye(n, dtype=bool)

        def tri(s, d2, d3):
            l1 = s - np.log(np.exp(s).sum())
            u2 = s[None, :] + d2 * same
            u2 = np.where(eye, -1e9, u2)
            l2 = u2 - np.log(np.exp(u2).sum(1, keepdims=True))
            u3 = s[None, None, :] + d3 * (same[:, None, :] | same[None, :, :])
            m3 = eye[:, None, :] | eye[None, :, :]
            u3 = np.where(m3, -1e9, u3)
            l3 = u3 - np.log(np.exp(u3).sum(2, keepdims=True))
            p = np.exp(l1[I] + l2[I, J] + l3[I, J, K])
            return p / p.sum()
        pv = tri(true_s, 0.9, 0.5)
        w = keys[rng.choice(len(keys), p=pv)]
        # 市場: 強い選手を買いすぎ、ライン連動を過小評価、ノイズあり
        mkt_s = (true_s * (1 + bias) + rng.normal(0, 0.25, n))
        qv = tri(mkt_s, 0.9 * (1 - bias), 0.5 * (1 - bias))
        odds = np.maximum(1.0, np.floor(0.75 / qv * 10) / 10)
        for (i, j, kk), o in zip(keys, odds):
            OD.append((rid, f"{cars[i]}-{cars[j]}-{cars[kk]}", float(o), "final"))
        wo = odds[keys.index(w)]
        RES.append(dict(race_id=rid, first=int(cars[w[0]]), second=int(cars[w[1]]),
                        third=int(cars[w[2]]), tri_payout=int(round(wo * 100))))
    return (pd.DataFrame(R), pd.DataFrame(E), pd.DataFrame(RES),
            pd.DataFrame(OD, columns=["race_id", "combo", "odds", "snapshot"]))
