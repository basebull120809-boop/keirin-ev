"""展開シナリオ予想：起こりうる展開を全部想定して3連単の確率を出す

  段1  主導権：レースに出ている全ラインについて「そのラインが主導権（最終バック＝B）を取る確率」
        材料はラインの先頭選手の全特徴量（B・S回数、逃げ・捲り、競走得点、記者の展開予想・印…）とライン人数
  段2  着順：主導権を取ったラインごとに、全504通りの着順の確率
        全特徴量に「主導権ラインの先頭・番手・3番手・単騎」「ほかのラインの先頭・番手・単騎」を足して学習
  段3  全展開を起こりやすさで混ぜる → P(3連単) = Σ_ライン P(主導権) × P(着順 | 主導権)
  段4  オッズ（市場の確率）と混ぜる。混ぜ具合は学習期間で決める（a=展開予想, b=市場）

自分の確率で自分を採点しない。成績は過去の本当の結果で確かめる（walk-forward）。
2026/07〜11 の 9,623 レース（毎月その前までで学習し直し）:
  3連単上位1点 的中13.6%・回収率86% ／ 3連複上位5点 的中64%・回収率80%（確定オッズ・100円均等）
  予想した的中数と実際の的中数はほぼ一致（例：3連複5点 予想5,992 → 実際5,998）
"""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .model import TrifectaModel, FEATURES, MAXN, _logsoftmax

PATH = "models/scenario.json"
N_EXTRA = 8
# 過去検証（walk-forward）の成績。表示に使う。数字は train で更新
BACKTEST = dict(period="2026-07〜2026-11", races=9409, trio5_hit=0.637, trio5_roi=0.797)


def lead_labels(root="data/daily"):
    """race_id → 最終バック（B）を取った車番"""
    L = []
    for f in Path(root).glob("*/*/*/finish.csv.gz"):
        try:
            L.append(pd.read_csv(f, dtype={"race_id": str}, usecols=["race_id", "car", "sb"]))
        except Exception:
            pass
    if not L:
        return {}
    fin = pd.concat(L)
    fin = fin[fin["sb"].fillna("").astype(str).str.upper().str.contains("B")]
    return fin.drop_duplicates("race_id").set_index("race_id")["car"].astype(int).to_dict()


def _idx():
    return [FEATURES.index(x) for x in ["is_head", "is_2nd", "is_3rd", "is_single"]]


def candidates(T):
    """各レースの候補ライン（最大 MAXN）: C=代表（先頭）の枠, CL=ラインid"""
    ih, _, _, isg = _idx()
    R = len(T.race_ids)
    C = np.full((R, MAXN), -1)
    CL = np.full((R, MAXN), -999)
    for r in range(R):
        seen = []
        for k in range(MAXN):
            if T.mask[r, k] and T.line[r, k] not in seen:
                seen.append(T.line[r, k])
        for c, l in enumerate(seen[:MAXN]):
            slots = [k for k in range(MAXN) if T.mask[r, k] and T.line[r, k] == l]
            hd = [k for k in slots if T.X[r, k, ih] > 0.5 or T.X[r, k, isg] > 0.5]
            C[r, c] = hd[0] if hd else slots[0]
            CL[r, c] = l
    return C, CL


def _headX(T, C, CL):
    R, F = len(T.race_ids), T.X.shape[2]
    Xh = np.zeros((R, MAXN, F + 1))
    for r in range(R):
        for c in range(MAXN):
            if C[r, c] >= 0:
                Xh[r, c, :F] = T.X[r, C[r, c]]
                Xh[r, c, F] = (T.line[r] == CL[r, c]).sum()
    return Xh


def _augX(T, C, CL, c_of):
    ih, i2, i3, isg = _idx()
    X = T.X
    R = len(T.race_ids)
    E = np.zeros((R, MAXN, N_EXTRA))
    for r in range(R):
        c = c_of[r]
        if c < 0 or C[r, c] < 0:
            continue
        inl = (T.line[r] == CL[r, c]).astype(float)
        out = 1 - inl
        hx, x2, x3, sg = X[r, :, ih], X[r, :, i2], X[r, :, i3], X[r, :, isg]
        E[r, :, 0] = inl; E[r, :, 1] = inl * hx; E[r, :, 2] = inl * x2; E[r, :, 3] = inl * x3
        E[r, :, 4] = inl * sg; E[r, :, 5] = out * hx; E[r, :, 6] = out * x2; E[r, :, 7] = out * sg
    return np.concatenate([X, E], 2)


def _with_X(T, X):
    t = copy.copy(T)
    t.X = X
    return t


def _take(T, idx):
    t = copy.copy(T)
    for a in ["X", "mask", "line", "cars", "same"]:
        setattr(t, a, getattr(T, a)[idx])
    t.race_ids = [T.race_ids[i] for i in idx]
    return t


class _Order(TrifectaModel):
    """着順の部分（特徴量の列数が FEATURES より多い版）"""
    def fit_x(self, T, first, second, third):
        R = len(first); rr = np.arange(R)
        self.beta = np.zeros(T.X.shape[2])

        def nll1(b):
            return -_logsoftmax(T.X @ b, T.mask)[rr, first].sum() / R + self.l2 * (b ** 2).sum()
        self.beta = minimize(nll1, self.beta, method="L-BFGS-B").x
        s = self.strength(T)

        def nll23(g):
            g2, d2, g3, d3 = g
            m2 = T.mask.copy(); m2[rr, first] = False
            l2 = _logsoftmax(g2 * s + d2 * T.same[rr, first], m2)[rr, second]
            m3 = m2.copy(); m3[rr, second] = False
            l3 = _logsoftmax(g3 * s + d3 * (T.same[rr, first] | T.same[rr, second]), m3)[rr, third]
            return -(l2 + l3).sum() / R
        self.g = minimize(nll23, self.g, method="L-BFGS-B").x
        return self


class ScenarioModel:
    def __init__(self, l2_lead=0.02):
        self.l2_lead = l2_lead
        self.b_lead = None
        self.order = _Order()
        self.blend = np.array([0.0, 1.0])

    # ---- 学習 ----
    def fit(self, T, first, second, third, Q, win_combo, lead_car):
        """lead_car: 各レースの B を取った車番（無ければ -1）"""
        C, CL = candidates(T)
        R = len(T.race_ids)
        lead = np.full(R, -1)
        for r in range(R):
            b = lead_car[r]
            if b >= 0 and b in T.cars[r]:
                k = int(np.where(T.cars[r] == b)[0][0])
                hit = np.where(CL[r] == T.line[r, k])[0]
                if len(hit):
                    lead[r] = int(hit[0])
        ok = np.where(lead >= 0)[0]
        Xh = _headX(T, C, CL)[ok]; m = (C >= 0)[ok]; y = lead[ok]; n = len(ok)
        F1 = Xh.shape[2]

        def nll(b):
            return -_logsoftmax(Xh @ b, m)[np.arange(n), y].mean() + self.l2_lead * (b ** 2).sum()
        self.b_lead = minimize(nll, np.zeros(F1), method="L-BFGS-B").x
        To = _take(T, ok)
        Co, CLo = C[ok], CL[ok]
        self.order.fit_x(_with_X(To, _augX(To, Co, CLo, y)), first[ok], second[ok], third[ok])
        P = self.mix(T)
        lp = np.log(P + 1e-12); lq = np.log(Q + 1e-12); v = Q > 0

        def nb(ab):
            return -_logsoftmax(ab[0] * lp + ab[1] * lq, v)[np.arange(R), win_combo].mean()
        self.blend = minimize(nb, np.array([0.5, 0.5]), method="L-BFGS-B", bounds=[(0, 3), (0, 3)]).x
        return self

    # ---- 予測 ----
    def lead_probs(self, T, C=None, CL=None):
        if C is None:
            C, CL = candidates(T)
        m = C >= 0
        return np.exp(_logsoftmax(_headX(T, C, CL) @ self.b_lead, m)) * m

    def mix(self, T):
        """展開を全部混ぜた確率（オッズとは混ぜない） (R,504)"""
        C, CL = candidates(T)
        PA = self.lead_probs(T, C, CL)
        R = len(T.race_ids)
        P = np.zeros((R, 504))
        for c in range(MAXN):
            w = PA[:, c]
            if not (w > 0).any():
                continue
            P += w[:, None] * self.order.combo_probs(_with_X(T, _augX(T, C, CL, np.full(R, c))))
        return P / np.maximum(P.sum(1, keepdims=True), 1e-300)

    def predict(self, T, Q):
        P = self.mix(T)
        v = Q > 0
        u = np.where(v, self.blend[0] * np.log(P + 1e-12) + self.blend[1] * np.log(Q + 1e-12), -1e9)
        u -= u.max(1, keepdims=True)
        p = np.where(v, np.exp(u), 0)
        return p / p.sum(1, keepdims=True)

    def lead_lines(self, T, r=0):
        """[(ラインの車番リスト, 主導権確率)]（確率の高い順）"""
        C, CL = candidates(_take(T, [r]))
        PA = self.lead_probs(_take(T, [r]), C, CL)[0]
        out = []
        for c in range(MAXN):
            if C[0, c] < 0:
                continue
            cars = [int(T.cars[r, k]) for k in range(MAXN) if T.mask[r, k] and T.line[r, k] == CL[0, c]]
            out.append((cars, float(PA[c])))
        return sorted(out, key=lambda x: -x[1])

    # ---- 保存 ----
    def to_json(self, path=PATH, **meta):
        d = dict(kind="scenario", l2_lead=self.l2_lead, b_lead=list(map(float, self.b_lead)),
                 beta=list(map(float, self.order.beta)), g=list(map(float, self.order.g)),
                 blend=list(map(float, self.blend)), features=FEATURES, **meta)
        Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, path=PATH):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("features") != FEATURES:
            raise ValueError("特徴量の構成が学習時と違うため使えません（再学習が必要）")
        m = cls(l2_lead=d["l2_lead"])
        m.b_lead = np.array(d["b_lead"]); m.order.beta = np.array(d["beta"])
        m.order.g = np.array(d["g"]); m.blend = np.array(d["blend"])
        m.meta = {k: v for k, v in d.items() if k not in ("b_lead", "beta", "g")}
        return m


def train_from(D, path=PATH, log=print):
    labels = lead_labels()
    T = D["T"]
    lead_car = np.array([labels.get(str(rid), -1) for rid in T.race_ids])
    m = ScenarioModel().fit(T, D["first"], D["second"], D["third"], D["Q"], D["win_combo"], lead_car)
    dates = D["races"]["date"]
    m.to_json(path, n_races=len(T.race_ids), n_lead=int((lead_car >= 0).sum()),
              period=f"{dates.min()}〜{dates.max()}",
              trained_at=f"{pd.Timestamp.now(tz='Asia/Tokyo'):%Y-%m-%d %H:%M}", backtest=BACKTEST)
    log(f"展開シナリオ予想 学習完了 {len(T.race_ids)}レース（主導権ラベル {int((lead_car >= 0).sum())}）"
        f" 混ぜ具合 展開={m.blend[0]:.3f} 市場={m.blend[1]:.3f}")
    return m


TRIO_POINTS = 5
TRIO_STAKE = 600

# ---- 展開ベースの買い目 ----
TENKAI_COVER = 0.75      # 主導権の確率をこの合計まで想定する
TENKAI_MAX_LINES = 3     # 想定する展開の数の上限
TENKAI_PER_LINE = 2      # 展開ごとの3連複の点数
TENKAI_STAKE = 500


def tenkai_trio(m, T, r, trio_odds=None, cover=TENKAI_COVER, max_lines=TENKAI_MAX_LINES,
                per_line=TENKAI_PER_LINE, stake=TENKAI_STAKE):
    """展開ごとに残りそうな3人を出す（オッズとは混ぜない。人気で目を選ばない）。
    主導権を取りそうなラインを確率の高い順に合計 cover まで（最大 max_lines 本）想定し、
    そのラインが主導権を取ったときの着順の確率から3連複を上から per_line 点ずつ。
    戻り値: 買い目リスト（scen=展開の説明, lead=主導権確率, pc=その展開での確率）, 展開リスト"""
    from .model import COMBOS
    Tr = _take(T, [r])
    C, CL = candidates(Tr)
    PA = m.lead_probs(Tr, C, CL)[0]
    order = [c for c in np.argsort(-PA) if C[0, c] >= 0 and PA[c] > 0]
    bets, scens, seen, cum = [], [], set(), 0.0
    for c in order[:max_lines]:
        line = [int(Tr.cars[0, k]) for k in range(MAXN) if Tr.mask[0, k] and Tr.line[0, k] == CL[0, c]]
        Pc = m.order.combo_probs(_with_X(Tr, _augX(Tr, C, CL, np.array([c]))))[0]
        pt = {}
        for n, (i, j, k) in enumerate(COMBOS):
            if Pc[n] > 0 and Tr.mask[0, i] and Tr.mask[0, j] and Tr.mask[0, k]:
                key = "=".join(map(str, sorted(int(Tr.cars[0, x]) for x in (i, j, k))))
                pt[key] = pt.get(key, 0.0) + float(Pc[n])
        ok = [x for x in sorted(pt, key=lambda x: -pt[x]) if trio_odds is None or float(trio_odds.get(x, 0) or 0) > 0]
        lab = "-".join(map(str, line)) + ("ライン" if len(line) > 1 else "（単騎）") + "が主導権"
        picks = []
        for key in ok:
            if len(picks) >= per_line:
                break
            if key in seen:
                continue
            seen.add(key); picks.append(key)
            o = float(trio_odds.get(key, 0)) if trio_odds else 0.0
            bets.append(dict(combo=key, stake=stake, odds=o, p=round(pt[key] * float(PA[c]), 5),
                             ev=round(pt[key] * float(PA[c]) * o, 3), scen=lab,
                             lead=round(float(PA[c]), 3), pc=round(pt[key], 3)))
        scens.append(dict(line=line, lead=float(PA[c]), label=lab, picks=picks))
        cum += float(PA[c])
        if cum >= cover:
            break
    return bets, scens


GUARD_DRIFT = 0.93       # 判定時のオッズは確定時に下がる（実測で約7%）。配分はこの分を見込んで安全側に組む
GUARD_UNIT = 100


GUARD_MIN_SYNTH = 1.7    # どの目が当たっても 投資額×この倍率 以上が戻る配分にできなければ見送り（config の synth_floor で上書き）


def guard_stakes(bets, budget=None, unit=GUARD_UNIT, drift=GUARD_DRIFT, min_synth=GUARD_MIN_SYNTH):
    """どの目が当たっても払戻が 投資額の合計×min_synth を下回らないように金額を配分し直す（トリガミ・薄すぎる払戻の防止）。
    各目の払戻（オッズ×drift で見込む）がそろうよう、オッズの低い目ほど厚く買う。予算 budget（既定は元の合計）の範囲で、
    そろえた払戻ができるだけ大きくなる配分を選ぶ。どう配分しても無理なら None（見送り）。"""
    if not bets:
        return None
    budget = budget or int(sum(b["stake"] for b in bets))
    od = [float(b.get("odds") or 0) * drift for b in bets]
    if min(od) <= min_synth or 1 / sum(1 / o for o in od) < min_synth:   # 合成オッズが下限未満なら、どう配っても届かない
        return None
    best = None
    T = unit
    while T <= budget * max(od):
        st = [int(-(-T // (o * unit))) * unit for o in od]   # 払戻 T 以上になる最小の金額（unit 単位）
        tot = sum(st)
        if tot > budget:
            break
        if min(s * o for s, o in zip(st, od)) >= tot * min_synth:
            best = st
        T += 10
    if best is None:
        return None
    out = []
    for b, s in zip(bets, best):
        b = dict(b, stake=s)
        out.append(b)
    return out


def prob_stakes(bets, budget, unit=GUARD_UNIT, drift=GUARD_DRIFT, min_back=1.0):
    """当たる確率に比例して厚く配る（2026-11-24）。ただしどの目が当たっても 払戻 ≥ 合計×min_back（トリガミなし）を守る。
    1) 各目に「当たれば合計 budget 以上が戻る」最低額を置く  2) 残りを確率に比例するよう unit ずつ足す。
    最低額だけで budget を超えるなら None。実績（11日・424レース、合成1.6倍以上）で回収率 73.8%→75.8%、当たる率は変わらず。"""
    if not bets:
        return None
    od = [float(b.get("odds") or 0) * drift for b in bets]
    if min(od) <= 0:
        return None
    ps = [max(float(b.get("p") or 0), 0.0) for b in bets]
    tp = sum(ps) or 1.0
    st = [int(-(-budget * min_back // (o * unit))) * unit for o in od]
    if sum(st) > budget:
        return None
    while sum(st) + unit <= budget:
        i = max(range(len(st)), key=lambda k: budget * ps[k] / tp - st[k])
        st[i] += unit
    tot = sum(st)
    if min(s * o for s, o in zip(st, od)) < tot * min_back:
        return None
    return [dict(b, stake=s) for b, s in zip(bets, st)]


def trio_top(P, cars, COMBOS, trio_odds, n=TRIO_POINTS, stake=TRIO_STAKE):
    """3連複を確率の高い順に n 点（オッズの付いている目だけ）。戻り値: 買い目リスト, 合計的中確率"""
    pt = {}
    for k, (i, j, l) in enumerate(COMBOS):
        if P[k] <= 0:
            continue
        key = "=".join(map(str, sorted([int(cars[i]), int(cars[j]), int(cars[l])])))
        pt[key] = pt.get(key, 0.0) + float(P[k])
    order = sorted((c for c in pt if trio_odds.get(c, 0) > 0), key=lambda c: -pt[c])[:n]
    bets = [dict(combo=c, stake=stake, odds=float(trio_odds[c]), p=round(pt[c], 5),
                 ev=round(pt[c] * float(trio_odds[c]), 3)) for c in order]
    return bets, float(sum(pt[c] for c in order))
