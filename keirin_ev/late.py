"""締切直前のオッズの記録と、判定時オッズからの下落の実測

判定（締切の約6分前）のあとも締切までにオッズは動き、当たった目ほど配当が下がる
（11/14〜23 のライブ記録では、的中した目の確定配当は判定時オッズの約0.8倍）。
締切 LATE_MAX〜LATE_MIN 分前にもう一度オッズを取り、data/live/日付/odds_late_*.csv.gz に保存する。
購入・判定には使わない（記録だけ）。`python main.py latedrift` で下落の大きさを集計する。
"""
import os
import re
from pathlib import Path
import pandas as pd

from . import kdreams

LIVE = Path("data/live")
LATE_MAX = float(os.environ.get("LATE_MAX", "1.2"))   # 締切何分前から直前オッズを取るか
LATE_MIN = float(os.environ.get("LATE_MIN", "0.2"))   # これより締切に近ければ取らない（締切後に取ると発売終了で読めない）
KINDS = ("trifecta", "exacta", "trio", "quinella")


def done_ids(date):
    """直前オッズを記録済みのレースID（実行し直したときの二重取得を防ぐ）"""
    f = LIVE / date / "odds_late_trio.csv.gz"
    if not f.exists():
        return set()
    try:
        return set(pd.read_csv(f, dtype={"race_id": str}, usecols=["race_id"])["race_id"])
    except Exception:
        return set()


def snapshot(date, venue, rid, dl=None):
    """締切直前のオッズを4券種まとめて保存する。保存した券種の数を返す"""
    html = kdreams.fetch(f"/{venue}/racedetail/{rid}/", refresh=True)
    t = pd.Timestamp.now(tz="Asia/Tokyo")
    sp = kdreams.soup(html)
    day = LIVE / date
    day.mkdir(parents=True, exist_ok=True)
    lead = round((dl - t).total_seconds() / 60, 2) if dl is not None else None
    n = 0
    for nm, fn in (("trifecta", kdreams.parse_odds), ("exacta", kdreams.parse_odds_exacta),
                   ("trio", kdreams.parse_odds_trio), ("quinella", kdreams.parse_odds_quinella)):
        try:
            df = fn(sp, rid)
        except Exception as e:
            print("直前オッズの読み取り失敗", rid, nm, e, flush=True)
            continue
        if df is None or not len(df):
            continue
        df = df[["race_id", "combo", "odds"]].assign(race_id=str(rid), taken_at=t.strftime("%H:%M:%S"), lead=lead)
        fo = day / f"odds_late_{nm}.csv.gz"
        df.to_csv(fo, mode="a", index=False, compression="gzip", header=not fo.exists())
        n += 1
    return n


# ---------- 集計：判定時オッズ → 直前オッズ → 確定配当 ----------
_TRIO = {"trio", "pool", "hit4", "evr3", "evr3x", "trio_z", "trio_m"}
_QN = {"qn", "evr2", "evr2x", "cmp2", "qn_z", "qn_m"}
_EX = {"ex", "tenkai", "ex_z", "tenkai_z", "ex_m"}


def _key(c, ordered):
    p = re.split("[-=]", str(c))
    return "-".join(p) if ordered else "=".join(sorted(p, key=int))


def drift(dates=None):
    """記録済みの日ごとに、買い目の判定時オッズ・直前オッズ・確定配当を突き合わせて表にする"""
    days = sorted(p.name for p in LIVE.glob("20*") if (p / "odds_late_trio.csv.gz").exists())
    if dates:
        days = [d for d in days if d in set(dates)]
    rows = []
    for d in days:
        day = LIVE / d
        if not (day / "bets.csv").exists():
            continue
        b = pd.read_csv(day / "bets.csv", dtype={"race_id": str})
        res = (pd.read_csv(day / "results.csv", dtype={"race_id": str}).drop_duplicates("race_id").set_index("race_id")
               if (day / "results.csv").exists() else pd.DataFrame())
        late = {}
        for nm, ordered in (("trio", 0), ("quinella", 0), ("exacta", 1)):
            f = day / f"odds_late_{nm}.csv.gz"
            if f.exists():
                x = pd.read_csv(f, dtype={"race_id": str}).drop_duplicates(["race_id", "combo"], keep="last")
                late[nm] = {(r, _key(c, ordered)): o for r, c, o in x[["race_id", "combo", "odds"]].values}
        for x in b.itertuples():
            if x.kind in _TRIO:
                nm, ordered, col = "trio", 0, ("trio_combo", "trio_payout")
            elif x.kind in _QN:
                nm, ordered, col = "quinella", 0, ("qn_combo", "qn_payout")
            elif x.kind in _EX:
                nm, ordered, col = "exacta", 1, ("ex_combo", "ex_payout")
            else:
                continue
            k = _key(x.combo, ordered)
            lo = late.get(nm, {}).get((x.race_id, k))
            hit, final = None, None
            if x.race_id in res.index:
                rr = res.loc[x.race_id]
                hit = _key(rr[col[0]], ordered) == k
                final = rr[col[1]] / 100 if hit else None
            rows.append(dict(date=d, kind=x.kind, odds=x.odds, late=lo, hit=hit, final=final, p=x.p, stake=x.stake))
    o = pd.DataFrame(rows)
    if o.empty:
        print("直前オッズの記録がまだありません（ライブ運用で溜まります）")
        return o
    o = o[o.late.notna() & (o.odds > 0)]
    print(f"突き合わせできた買い目 {len(o)}点（{o.date.nunique()}日）")
    print("■ 判定時 → 直前のオッズ比（全買い目）  中央値", round((o.late / o.odds).median(), 3))
    h = o[o.hit == True]   # noqa: E712
    if len(h):
        print(f"■ 的中 {len(h)}本")
        print("  判定時 → 直前  ", round((h.late / h.odds).median(), 3))
        print("  直前   → 確定  ", round((h.final / h.late).median(), 3))
        print("  判定時 → 確定  ", round((h.final / h.odds).median(), 3))
    s = o[o.hit.notna()]
    if len(s):
        ret = (s.final.fillna(0) * s.stake).sum()
        print("■ 回収率  確定配当", round(ret / s.stake.sum(), 3))
        for nm, col in (("判定時オッズで期待値1以上", "odds"), ("直前オッズで期待値1以上", "late")):
            m = s[s.p * s[col] >= 1]
            if len(m):
                print(f"  {nm}: {len(m)}点 回収率 {round((m.final.fillna(0) * m.stake).sum() / m.stake.sum(), 3)}")
    return o
