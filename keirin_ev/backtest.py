"""ウォークフォワード検証と市場効率レポート"""
import numpy as np
import pandas as pd
from .model import TrifectaModel, build_tensors, RaceTensor, COMBOS
from .strategy import Rules, pick_bets


def _sub(D, idx):
    T = D["T"]
    S = RaceTensor.__new__(RaceTensor)
    S.race_ids = [T.race_ids[i] for i in idx]
    for a in ["X", "mask", "line", "cars", "same"]:
        setattr(S, a, getattr(T, a)[idx])
    out = dict(T=S)
    for k in ["Q", "O", "first", "second", "third", "win_combo", "payout"]:
        if k in D:
            out[k] = D[k][idx]
    out["races"] = D["races"].iloc[idx]
    return out


def walk_forward(con, rules=Rules(), bankroll0=300_000, min_train_months=6,
                 compound=False):
    D = build_tensors(con)
    races = D["races"].reset_index(drop=True)
    months = pd.to_datetime(races["date"]).dt.to_period("M")
    ulist = sorted(months.unique())
    bank = bankroll0
    logs, coefs = [], []
    for m in ulist[min_train_months:]:
        tr = np.where(months < m)[0]
        te = np.where(months == m)[0]
        A, B = _sub(D, tr), _sub(D, te)
        mdl = TrifectaModel().fit(A["T"], A["first"], A["second"], A["third"],
                                  A["Q"], A["win_combo"])
        coefs.append((str(m), *mdl.blend))
        P = mdl.predict(B["T"], B["Q"])
        for r in range(len(te)):
            base = bank if compound else bankroll0
            for n, st, p, o, ev in pick_bets(P[r], B["O"][r], base, rules):
                hit = n == B["win_combo"][r]
                pay = st / 100 * B["payout"][r] if hit else 0
                bank += pay - st
                i, j, k = COMBOS[n]
                cars = B["T"].cars[r]
                logs.append(dict(month=str(m), race_id=B["T"].race_ids[r],
                                 combo=f"{cars[i]}-{cars[j]}-{cars[k]}",
                                 stake=st, odds=o, p=p, ev=ev, hit=hit,
                                 payout=pay, bank=bank))
    log = pd.DataFrame(logs)
    return log, pd.DataFrame(coefs, columns=["month", "a_model", "b_market"])


def summarize(log, bankroll0=300_000, n_boot=2000, seed=0):
    if log.empty:
        return {"bets": 0}
    st, pay = log["stake"].sum(), log["payout"].sum()
    eq = bankroll0 + (log["payout"] - log["stake"]).cumsum()
    dd = (eq / eq.cummax() - 1).min()
    # レース単位ブートストラップで回収率の95%区間
    g = log.groupby("race_id")[["stake", "payout"]].sum().values
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        s = g[rng.integers(0, len(g), len(g))]
        bs.append(s[:, 1].sum() / s[:, 0].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"購入点数": len(log), "購入レース数": log["race_id"].nunique(),
            "的中数": int(log["hit"].sum()), "投資額": int(st), "払戻額": int(pay),
            "回収率": round(pay / st, 3), "回収率95%区間": f"{lo:.3f} 〜 {hi:.3f}",
            "最大ドローダウン": round(dd, 3), "最終資金": int(eq.iloc[-1])}


def market_report(con):
    """人気順・オッズ帯ごとの回収率（100円均等買いした場合）"""
    D = build_tensors(con)
    O, W = D["O"], D["win_combo"]
    rows = []
    for r in range(O.shape[0]):
        o = O[r]; valid = np.where(o > 0)[0]
        order = valid[np.argsort(o[valid])]
        for rank, n in enumerate(order, 1):
            rows.append((rank, o[n], n == W[r]))
    df = pd.DataFrame(rows, columns=["rank", "odds", "hit"])
    df["ret"] = np.where(df["hit"], df["odds"], 0)
    df["人気帯"] = pd.cut(df["rank"], [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 504],
                       labels=["1", "2", "3", "4-5", "6-10", "11-20", "21-50",
                               "51-100", "101-200", "201-"])
    df["オッズ帯"] = pd.cut(df["odds"], [0, 5, 10, 20, 50, 100, 300, 1000, 1e9],
                        labels=["~5", "5-10", "10-20", "20-50", "50-100",
                                "100-300", "300-1000", "1000~"])
    f = lambda k: df.groupby(k, observed=True).agg(
        点数=("hit", "size"), 的中率=("hit", "mean"), 回収率=("ret", "mean")).round(4)
    return f("人気帯"), f("オッズ帯")
