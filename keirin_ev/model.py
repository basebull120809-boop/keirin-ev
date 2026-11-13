"""確率モデル

3段構成:
  1) 1着モデル   : 条件付きロジット。選手特徴量 + 市場が見ている1着確率(log)
  2) 2・3着モデル: Plackett-Luce型。1着(・2着)と同ラインの選手に連動ボーナス
  3) 市場ブレンド: log p = a*log p_model + b*log q_market を実データで推定
                   （= 市場を事前分布にしたベイズ更新に相当。a≈0 なら優位性なし）
最終的に 3連単全組み合わせの確率を返す。
"""
from itertools import permutations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .perspectives import P_FEATURES, add as add_perspectives

MAXN = 9
COMBOS = np.array(list(permutations(range(MAXN), 3)))   # 504通り (0始まり)
TAKE = 0.75

FEATURES = [
    "z_score", "z_win", "z_ren", "is_head", "is_2nd", "is_3rd",
    "is_single", "line_size", "head_x_short", "head_x_wind",
    "style_nige", "style_oi", "log_mkt_win",
] + P_FEATURES


# ---------- 特徴量 ----------
def _z(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0


def build_features(entries: pd.DataFrame, races: pd.DataFrame,
                   mkt_win: pd.DataFrame, history=None) -> pd.DataFrame:
    df = add_perspectives(entries, races, history)       # S・B・番手巧者・上がり・調子・地元など
    races = races.assign(race_id=races["race_id"].astype(str))
    mkt_win = mkt_win.assign(race_id=mkt_win["race_id"].astype(str))
    df = df.merge(races[["race_id", "bank_len", "wind"]], on="race_id", how="left")
    df = df.merge(mkt_win, on=["race_id", "car"], how="left")
    g = df.groupby("race_id")
    df["z_score"] = g["score"].transform(_z).fillna(0)
    df["z_win"] = g["win_rate"].transform(_z).fillna(0)
    df["z_ren"] = g["ren_rate"].transform(_z).fillna(0)
    df["is_head"] = (df["line_pos"] == 1) & (df["line_size"] > 1)
    df["is_2nd"] = df["line_pos"] == 2
    df["is_3rd"] = df["line_pos"] >= 3
    df["is_single"] = df["line_size"].fillna(1) <= 1
    df["line_size"] = df["line_size"].fillna(1)
    df["head_x_short"] = df["is_head"] & (df["bank_len"].fillna(400) <= 335)
    df["head_x_wind"] = df["is_head"] * df["wind"].fillna(0)
    df["style_nige"] = df["style"].fillna("").str.contains("逃")
    df["style_oi"] = df["style"].fillna("").str.contains("追")
    df["log_mkt_win"] = np.log(df["mkt_win"].clip(lower=1e-4)).fillna(np.log(1 / 9))
    for c in FEATURES:
        df[c] = df[c].astype(float)
    return df


def market_probs(odds: pd.DataFrame):
    """3連単オッズ → 市場の組み合わせ確率 q と 1着確率"""
    o = odds[odds["snapshot"] == "final"].copy()
    o["inv"] = 1.0 / o["odds"]
    o["q"] = o["inv"] / o.groupby("race_id")["inv"].transform("sum")
    o["first"] = o["combo"].str.split("-").str[0].astype(int)
    win = o.groupby(["race_id", "first"])["q"].sum().reset_index()
    win.columns = ["race_id", "car", "mkt_win"]
    return o[["race_id", "combo", "odds", "q"]], win


# ---------- テンソル化 ----------
class RaceTensor:
    def __init__(self, feat: pd.DataFrame, race_ids):
        self.race_ids = list(race_ids)
        R = len(self.race_ids)
        idx = {r: i for i, r in enumerate(self.race_ids)}
        self.X = np.zeros((R, MAXN, len(FEATURES)))
        self.mask = np.zeros((R, MAXN), bool)
        self.line = np.full((R, MAXN), -1, int)
        self.cars = np.zeros((R, MAXN), int)
        f = feat[feat["race_id"].isin(idx)].sort_values(["race_id", "car"])
        for rid, grp in f.groupby("race_id"):
            r = idx[rid]
            n = min(len(grp), MAXN)
            self.X[r, :n] = grp[FEATURES].values[:n]
            self.mask[r, :n] = True
            self.line[r, :n] = grp["line_id"].fillna(-1).astype(int).values[:n] + 0
            self.cars[r, :n] = grp["car"].values[:n]
        # 単騎は互いに別ライン扱い
        single = feat.set_index(["race_id", "car"])["is_single"]
        for r, rid in enumerate(self.race_ids):
            for k in range(MAXN):
                if self.mask[r, k] and single.get((rid, self.cars[r, k]), 0):
                    self.line[r, k] = 1000 + k
        same = (self.line[:, :, None] == self.line[:, None, :]) & (self.line[:, :, None] >= 0)
        self.same = same & self.mask[:, :, None] & self.mask[:, None, :]
        d = np.arange(MAXN)
        self.same[:, d, d] = False

    def pos(self, rid, car):
        r = self.race_ids.index(rid)
        return int(np.where(self.cars[r] == car)[0][0])


def _logsoftmax(u, mask, axis=-1):
    u = np.where(mask, u, -1e9)
    m = u.max(axis=axis, keepdims=True)
    return u - m - np.log(np.exp(u - m).sum(axis=axis, keepdims=True))


# ---------- モデル本体 ----------
class TrifectaModel:
    def __init__(self, l2=0.05, l2_view=None):
        self.l2 = l2
        self.l2_view = l2_view          # 視点（p_*）の縮め具合。None ならデータ量に応じて自動
        self.beta = np.zeros(len(FEATURES))
        self.g = np.array([1.0, 0.5, 1.0, 0.3])   # g2, d2, g3, d3
        self.blend = np.array([1.0, 0.0])         # a(モデル), b(市場)

    # 1着
    def strength(self, T):
        return T.X @ self.beta

    def fit(self, T: RaceTensor, first, second, third, q_tensor=None, win_combo=None):
        R = len(T.race_ids)
        rr = np.arange(R)
        # 市場の1着確率は縮めない。視点はデータが少ないうちは強めに縮める（約5,000レースで基本と同じ強さ）
        lv = self.l2_view if self.l2_view is not None else self.l2 * max(1.0, 5000 / max(R, 1))
        pen = np.array([0.0 if f == "log_mkt_win" else
                        np.sqrt(lv / self.l2) if f.startswith("p_") else 1.0 for f in FEATURES])

        def nll1(b):
            s = T.X @ b
            ls = _logsoftmax(s, T.mask)
            return -ls[rr, first].sum() / R + self.l2 * ((b * pen) ** 2).sum()
        self.beta = minimize(nll1, self.beta, method="L-BFGS-B").x
        s = self.strength(T)

        def nll23(g):
            g2, d2, g3, d3 = g
            m2 = T.mask.copy(); m2[rr, first] = False
            u2 = g2 * s + d2 * T.same[rr, first]
            l2 = _logsoftmax(u2, m2)[rr, second]
            m3 = m2.copy(); m3[rr, second] = False
            u3 = g3 * s + d3 * (T.same[rr, first] | T.same[rr, second])
            l3 = _logsoftmax(u3, m3)[rr, third]
            return -(l2 + l3).sum() / R
        self.g = minimize(nll23, self.g, method="L-BFGS-B").x

        if q_tensor is not None:
            lp = np.log(self.combo_probs(T) + 1e-12)
            lq = np.log(q_tensor + 1e-12)
            valid = q_tensor > 0

            def nllb(ab):
                u = ab[0] * lp + ab[1] * lq
                ls = _logsoftmax(u, valid)
                return -ls[rr, win_combo].sum() / R
            self.blend = minimize(nllb, np.array([0.5, 0.5]), method="L-BFGS-B",
                                  bounds=[(0, 3), (0, 3)]).x
        return self

    def combo_probs(self, T):
        """モデル単体の3連単確率 (R, 504)"""
        s = self.strength(T)
        g2, d2, g3, d3 = self.g
        R = s.shape[0]
        l1 = _logsoftmax(s, T.mask)                                  # (R,i)
        m2 = T.mask[:, None, :] & ~np.eye(MAXN, dtype=bool)[None]   # (R,i,j)
        u2 = g2 * s[:, None, :] + d2 * T.same                       # (R,i,j)
        l2 = _logsoftmax(u2, m2)
        eye = np.eye(MAXN, dtype=bool)
        m3 = (T.mask[:, None, None, :] & ~eye[None, :, None, :] & ~eye[None, None, :, :])
        sameik = T.same[:, :, None, :]
        samejk = T.same[:, None, :, :]
        u3 = g3 * s[:, None, None, :] + d3 * (sameik | samejk)
        l3 = _logsoftmax(np.broadcast_to(u3, (R, MAXN, MAXN, MAXN)), m3)
        i, j, k = COMBOS.T
        lp = l1[:, i] + l2[:, i, j] + l3[:, i, j, k]
        valid = T.mask[:, i] & T.mask[:, j] & T.mask[:, k]
        p = np.where(valid, np.exp(lp), 0)
        return p / p.sum(1, keepdims=True)

    def predict(self, T, q_tensor):
        """市場ブレンド後の最終確率 (R, 504)"""
        lp = np.log(self.combo_probs(T) + 1e-12)
        valid = q_tensor > 0
        a, b = self.blend
        u = np.where(valid, a * lp + b * np.log(q_tensor + 1e-12), -1e9)
        u -= u.max(1, keepdims=True)
        p = np.where(valid, np.exp(u), 0)
        return p / p.sum(1, keepdims=True)

    def to_json(self, path, **meta):
        import json
        d = dict(features=FEATURES, beta=list(self.beta), g=list(self.g),
                 blend=list(self.blend), **meta)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, path):
        import json
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d["features"] != FEATURES:
            raise ValueError("特徴量の定義が変わっています。再学習してください")
        m = cls()
        m.beta, m.g, m.blend = (np.array(d[k]) for k in ("beta", "g", "blend"))
        m.meta = d
        return m

    def coef_table(self):
        t = pd.Series(self.beta, index=FEATURES).round(3).to_frame("1着係数")
        extra = pd.Series({"2着:強さ": self.g[0], "2着:同ライン": self.g[1],
                           "3着:強さ": self.g[2], "3着:同ライン": self.g[3],
                           "ブレンド:モデル a": self.blend[0],
                           "ブレンド:市場 b": self.blend[1]}).round(3)
        return t, extra


# ---------- データ→テンソル ----------
def combo_index(T, rid, combo_str):
    r = T.race_ids.index(rid)
    cars = [int(c) for c in combo_str.split("-")]
    pos = [int(np.where(T.cars[r] == c)[0][0]) for c in cars]
    return pos


def build_tensors(con, race_ids=None, need_result=True, history=None):
    races = pd.read_sql("SELECT * FROM races", con)
    entries = pd.read_sql("SELECT * FROM entries", con)
    odds = pd.read_sql("SELECT * FROM odds_trifecta", con)
    res = pd.read_sql("SELECT * FROM results", con)
    q, win = market_probs(odds)
    ok = set(q["race_id"])
    ok -= set(entries.loc[entries["class"] == "L1", "race_id"])   # ガールズケイリンは対象外
    if need_result:
        ok &= set(res.dropna(subset=["first", "second", "third"])["race_id"])
    if race_ids is not None:
        ok &= set(race_ids)
    races = races[races["race_id"].isin(ok)].sort_values(["date", "race_id"])
    rids = races["race_id"].tolist()
    feat = build_features(entries[entries["race_id"].isin(ok)], races, win, history)
    T = RaceTensor(feat, rids)
    R = len(rids)
    # 市場確率・オッズを (R,504) に
    pos_of = {}
    for r in range(R):
        for k in range(MAXN):
            if T.mask[r, k]:
                pos_of[(rids[r], int(T.cars[r, k]))] = k
    code = {tuple(c): n for n, c in enumerate(COMBOS)}
    Q = np.zeros((R, len(COMBOS))); O = np.zeros((R, len(COMBOS)))
    ridx = {r: i for i, r in enumerate(rids)}
    qq = q[q["race_id"].isin(ridx)]
    for rid, combo, od, qv in qq[["race_id", "combo", "odds", "q"]].itertuples(index=False):
        cs = [int(c) for c in combo.split("-")]
        try:
            n = code[tuple(pos_of[(rid, c)] for c in cs)]
        except KeyError:
            continue
        Q[ridx[rid], n] = qv; O[ridx[rid], n] = od
    out = dict(T=T, Q=Q, O=O, races=races)
    if need_result:
        rs = res.set_index("race_id").loc[rids]
        f = np.array([pos_of[(r, int(c))] for r, c in zip(rids, rs["first"])])
        s = np.array([pos_of[(r, int(c))] for r, c in zip(rids, rs["second"])])
        t = np.array([pos_of[(r, int(c))] for r, c in zip(rids, rs["third"])])
        out.update(first=f, second=s, third=t,
                   win_combo=np.array([code[(a, b, c)] for a, b, c in zip(f, s, t)]),
                   payout=rs["tri_payout"].values.astype(float))
    return out
