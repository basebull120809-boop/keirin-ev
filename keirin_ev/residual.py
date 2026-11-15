"""市場との差を直接学ぶ3連単モデル（残差モデル）

  u(i,j,k) = b1·log q + b2·(log q)² + x_i·β1 + x_j·β2 + x_k·β3 + d12·同ライン(i,j) + d3·同ライン(k, i or j)
  P = softmax(u)（そのレースの全組み合わせで）

  q はオッズから見た市場確率。b2 は「人気薄ほど過大評価される」など人気による歪みを拾う。
  β は「オッズに対して来すぎ・来なさすぎ」の選手の特徴を拾う。何も無ければ β=0, b1=1, b2=0 で市場そのもの。
"""
import numpy as np
from scipy.optimize import minimize
from .model import COMBOS, MAXN, FEATURES

I, J, K = COMBOS.T
S1 = np.zeros((len(COMBOS), MAXN)); S1[np.arange(len(COMBOS)), I] = 1
S2 = np.zeros((len(COMBOS), MAXN)); S2[np.arange(len(COMBOS)), J] = 1
S3 = np.zeros((len(COMBOS), MAXN)); S3[np.arange(len(COMBOS)), K] = 1


class ResidualModel:
    def __init__(self, l2=0.02, features=None, lq_terms=2):
        self.l2 = l2
        self.features = [f for f in (features or FEATURES) if f != "log_mkt_win"]
        self.fidx = [FEATURES.index(f) for f in self.features]
        self.lq_terms = lq_terms
        self.theta = None

    def _parts(self, T, Q):
        X = T.X[:, :, self.fidx]
        X = np.where(T.mask[:, :, None], X, 0.0)
        valid = Q > 0
        lq = np.log(np.where(valid, Q, 1.0))
        lqc = lq + 6.0                                   # 中心をずらして2乗項を安定させる
        same12 = T.same[:, I, J].astype(float)
        same3 = (T.same[:, I, K] | T.same[:, J, K]).astype(float)
        return X, valid, lq, lqc, same12, same3

    def _u(self, th, X, lq, lqc, same12, same3):
        F = X.shape[2]
        b = th[:2]; d = th[2:4]; B = th[4:].reshape(3, F)
        s = X @ B.T                                      # (R,9,3)
        u = b[0] * lq + b[1] * lqc ** 2 * (self.lq_terms > 1) + s[:, I, 0] + s[:, J, 1] + s[:, K, 2] \
            + d[0] * same12 + d[1] * same3
        return u

    def fit(self, T, Q, win, x0=None):
        X, valid, lq, lqc, same12, same3 = self._parts(T, Q)
        R, _, F = X.shape
        rr = np.arange(R)
        wi, wj, wk = I[win], J[win], K[win]
        th0 = np.zeros(4 + 3 * F); th0[0] = 1.0
        if x0 is not None:
            th0 = x0

        def f(th):
            u = self._u(th, X, lq, lqc, same12, same3)
            u = np.where(valid, u, -1e9)
            m = u.max(1, keepdims=True)
            e = np.exp(u - m); Z = e.sum(1, keepdims=True); P = e / Z
            ll = (u[rr, win] - m[:, 0] - np.log(Z[:, 0])).sum()
            g = np.zeros_like(th)
            g[0] = (lq[rr, win] - (P * lq).sum(1)).sum()
            g[1] = (lqc[rr, win] ** 2 - (P * lqc ** 2).sum(1)).sum() if self.lq_terms > 1 else 0
            g[2] = (same12[rr, win] - (P * same12).sum(1)).sum()
            g[3] = (same3[rr, win] - (P * same3).sum(1)).sum()
            for p, (S, w) in enumerate([(S1, wi), (S2, wj), (S3, wk)]):
                M = P @ S                                # (R,9) 位置ごとの周辺確率
                Y = np.zeros_like(M); Y[rr, w] = 1
                g[4 + p * F: 4 + (p + 1) * F] = np.einsum("ri,rif->f", Y - M, X)
            reg = th.copy(); reg[0] -= 1.0; reg[:2] = 0          # 市場の項は縮めない
            loss = -ll / R + self.l2 * (reg ** 2).sum()
            grad = -g / R + 2 * self.l2 * reg
            return loss, grad
        self.theta = minimize(f, th0, jac=True, method="L-BFGS-B").x
        return self

    def predict(self, T, Q):
        X, valid, lq, lqc, same12, same3 = self._parts(T, Q)
        u = np.where(valid, self._u(self.theta, X, lq, lqc, same12, same3), -1e9)
        u -= u.max(1, keepdims=True)
        p = np.where(valid, np.exp(u), 0)
        return p / p.sum(1, keepdims=True)


    def to_json(self, path, **meta):
        import json
        from pathlib import Path
        d = dict(kind="residual", features=self.features, theta=list(map(float, self.theta)),
                 l2=self.l2, lq_terms=self.lq_terms, **meta)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, path):
        import json
        from pathlib import Path
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls(l2=d["l2"], features=d["features"], lq_terms=d.get("lq_terms", 2))
        if m.features != d["features"]:
            raise ValueError("特徴量の定義が変わっています。再学習してください")
        m.theta = np.array(d["theta"]); m.meta = d
        return m
