"""学習に使っていない期間で「市場（オッズ）より当てられているか」を測る。

  分け方 : 日付順に並べ、前の train_frac を学習、残りを検証（時系列なので混ぜない）
  指標   : 検証レースの「実際に来た3連単」の対数尤度。市場 q と比べた差（1レースあたり）
           ＞0 ならオッズより良く当てている。標準誤差つき
  回収   : 確定オッズで、モデル確率×オッズ（期待値）が閾値以上の目を100円ずつ買った場合
"""
import numpy as np
import pandas as pd
from .backtest import _sub


def split_idx(D, train_frac=0.7):
    dates = pd.to_datetime(D["races"]["date"].values)
    ud = np.sort(np.unique(dates))
    cut = ud[int(len(ud) * train_frac)]
    return np.where(dates < cut)[0], np.where(dates >= cut)[0], str(pd.Timestamp(cut).date())


def loglik_gain(P, Q, win):
    r = np.arange(len(win))
    lp = np.log(np.clip(P[r, win], 1e-12, None)); lq = np.log(np.clip(Q[r, win], 1e-12, None))
    d = lp - lq
    return dict(races=len(win), model_ll=float(lp.mean()), market_ll=float(lq.mean()),
                gain=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))


def roi_by_ev(P, O, win, payout, thresholds=(1.0, 1.1, 1.2, 1.3, 1.5)):
    out = []
    ev = P * O
    hitpay = np.zeros_like(O)
    hitpay[np.arange(len(win)), win] = payout       # 100円あたり払戻
    for t in thresholds:
        m = (ev >= t) & (O > 0)
        n = int(m.sum())
        st = n * 100.0
        pay = float(hitpay[m].sum())
        races = int(m.any(1).sum())
        # レース単位ブートストラップの95%区間
        rs = m.sum(1) * 100.0; rp = (hitpay * m).sum(1)
        idx = np.where(rs > 0)[0]
        lo = hi = np.nan
        if len(idx) > 20:
            rng = np.random.default_rng(0); bs = []
            for _ in range(1000):
                k = idx[rng.integers(0, len(idx), len(idx))]
                bs.append(rp[k].sum() / rs[k].sum())
            lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append(dict(ev_min=t, races=races, points=n, roi=pay / st if st else np.nan, lo=lo, hi=hi))
    return pd.DataFrame(out)


def evaluate(D, fit_predict, train_frac=0.7, label=""):
    tr, te, cut = split_idx(D, train_frac)
    A, B = _sub(D, tr), _sub(D, te)
    P = fit_predict(A, B)
    g = loglik_gain(P, B["Q"], B["win_combo"])
    g.update(label=label, cut=cut, train=len(tr))
    return g, roi_by_ev(P, B["O"], B["win_combo"], B["payout"]), P
