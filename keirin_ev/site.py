"""サイト（Claudeのページ）に流すデータを site/dashboard.json に書き出す"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("site/dashboard.json")
TARGET_FIRST = "2024-11-01"


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (o != o) else round(float(o), 4)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def load():
    try:
        return json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save(d):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d["updated_at"] = f"{pd.Timestamp.now(tz='Asia/Tokyo'):%Y-%m-%d %H:%M}"
    OUT.write_text(json.dumps(_clean(d), ensure_ascii=False, indent=1), encoding="utf-8")


def picks_and_paper():
    """当日（最新日）の買い目と、紙上運用の通算成績"""
    from .live import paper_results
    from .kdreams import VENUE_JA
    pb = paper_results()
    if pb is None:
        return None, None
    pb["race_id"] = pb["race_id"].astype(str)
    last = pb["date"].max()
    races = []
    for rid, g in pb[pb["date"] == last].groupby("race_id", sort=False):
        r = g.iloc[0]
        res = None
        if isinstance(r.get("win"), str):
            res = dict(combo=r["win"], payout=int(r["tri_payout"]) if r["tri_payout"] == r["tri_payout"] else None)
        races.append(dict(venue=VENUE_JA.get(r["venue"], r["venue"]), race_no=int(r["race_no"]),
                          deadline=str(r.get("deadline", "") or ""), judged_at=str(r["taken_at"]),
                          result=res,
                          bets=[dict(combo=b.combo, stake=int(b.stake), odds=float(b.odds),
                                     p=float(b.p), ev=float(b.ev),
                                     hit=None if b.hit != b.hit else bool(b.hit))
                                for b in g.itertuples()]))
    races.sort(key=lambda x: x["deadline"] or x["judged_at"])
    picks = dict(date=last, races=races)
    done = pb.dropna(subset=["hit"])
    paper = None
    if len(done):
        d = done.groupby("date").agg(stake=("stake", "sum"), payout=("payout", "sum"),
                                     points=("combo", "size"), hits=("hit", "sum")).reset_index()
        d["roi"] = d["payout"] / d["stake"]
        d["cum"] = (d["payout"] - d["stake"]).cumsum()
        st, py = done["stake"].sum(), done["payout"].sum()
        paper = dict(total=dict(stake=st, payout=py, roi=py / st, points=len(done),
                                hits=int(done["hit"].sum()), races=done["race_id"].nunique(),
                                days=len(d)),
                     daily=d.to_dict("records"))
    return picks, paper


def export_live():
    d = load()
    picks, paper = picks_and_paper()
    if picks:
        d["picks"] = picks
    if paper:
        d["paper"] = paper
    save(d)


def export_report(con, days, dates, market=None, backtest=None):
    import json as _j
    d = load()
    nr = con.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    months = con.execute("SELECT COUNT(DISTINCT substr(date,1,7)) FROM races").fetchone()[0]
    d["collect"] = dict(days=days, races=nr, months=months,
                        first=dates[0] if dates else None, last=dates[-1] if dates else None,
                        target_first=TARGET_FIRST)
    try:
        cfg = _j.loads(Path("config.json").read_text(encoding="utf-8"))
        d["config"] = dict(mode=cfg.get("mode"), bankroll=cfg.get("bankroll"),
                           ev_min=cfg.get("rules", {}).get("ev_min"))
    except Exception:
        pass
    try:
        m = _j.loads(Path("models/latest.json").read_text(encoding="utf-8"))
        d["model"] = dict(n_races=m.get("n_races"), period=m.get("period"),
                          trained_at=m.get("trained_at"), a=m["blend"][0], b=m["blend"][1])
    except Exception:
        pass
    if market is not None:
        r1, r2 = market
        d["market"] = dict(
            popularity=[dict(band=str(k), n=int(v["点数"]), hit=v["的中率"], roi=v["回収率"])
                        for k, v in r1.iterrows()],
            oddsband=[dict(band=str(k), n=int(v["点数"]), hit=v["的中率"], roi=v["回収率"])
                      for k, v in r2.iterrows()])
    if backtest is not None:
        d["backtest"] = backtest
    picks, paper = picks_and_paper()
    if picks:
        d["picks"] = picks
    if paper:
        d["paper"] = paper
    save(d)
