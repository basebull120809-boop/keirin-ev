"""当日運用：締切の数分前にオッズを取り直して期待値判定 → GitHubのIssueに通知

流れ
  1) 起動時に当日の全レースの締切時刻を取得
  2) 30秒ごとに見回り、締切まで LEAD_MAX〜LEAD_MIN 分のレースを判定
  3) 判定結果は data/live/日付/ に保存（オッズのスナップショット・買い目）
  4) 買い目があれば当日のIssueにコメント（GitHubアプリに通知が届く）

実際の購入はしない。通知を見て、買うかどうかは自分で決める。
"""
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

from . import kdreams
from .db import connect
from .collector import store
from .model import TrifectaModel, build_tensors, COMBOS
from .strategy import Rules, pick_bets

JST = "Asia/Tokyo"
LIVE = Path("data/live")
LEAD_MAX = float(os.environ.get("LEAD_MAX", "6"))   # 締切何分前から判定するか
LEAD_MIN = float(os.environ.get("LEAD_MIN", "1.5"))  # これより締切に近ければ見送り


def now():
    return pd.Timestamp.now(tz=JST)


def load_config():
    p = Path("config.json")
    cfg = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    rules = Rules(**{k: v for k, v in cfg.get("rules", {}).items()
                     if k in Rules.__dataclass_fields__})
    return cfg, rules


def load_invest_rules(cfg):
    """投資判定（期待値で勝てる目だけ）のルール。設定が無ければ使わない"""
    r = cfg.get("invest_rules")
    if not r:
        return None
    return Rules(**{k: v for k, v in r.items() if k in Rules.__dataclass_fields__})


# ---------- 通知（GitHub Issue） ----------
class Notifier:
    def __init__(self, date):
        self.repo = os.environ.get("GITHUB_REPOSITORY")
        self.token = os.environ.get("GITHUB_TOKEN")
        self.title = f"買い目 {date}"
        self.num = None

    def _api(self, method, path, **kw):
        import requests
        r = requests.request(method, f"https://api.github.com/repos/{self.repo}{path}",
                             headers={"Authorization": f"Bearer {self.token}",
                                      "Accept": "application/vnd.github+json"},
                             timeout=20, **kw)
        r.raise_for_status()
        return r.json()

    def post(self, text):
        print(text, flush=True)
        if not (self.repo and self.token):
            return
        try:
            if self.num is None:
                for it in self._api("GET", "/issues", params={"state": "open", "per_page": 100}):
                    if it["title"] == self.title:
                        self.num = it["number"]
                if self.num is None:
                    self.num = self._api("POST", "/issues", json={
                        "title": self.title,
                        "body": "締切前オッズで期待値判定した買い目をここに流します。"})["number"]
            self._api("POST", f"/issues/{self.num}/comments", json={"body": text})
        except Exception as e:
            print("通知失敗:", e, flush=True)


# ---------- 判定 ----------
def deadline_ts(date, hm):
    h, m = hm
    return pd.Timestamp(f"{date} 00:00", tz=JST) + pd.Timedelta(hours=h, minutes=m)


def schedule(date):
    """当日の [(締切, 場, レースID)]"""
    out = []
    for venue, rid in kdreams.race_ids_of_day(date):
        try:
            s = kdreams.soup(kdreams.fetch(f"/{venue}/racedetail/{rid}/", refresh=True))
            hm = kdreams.parse_deadline(s)
            ent = kdreams.parse_entries(s, rid)
        except Exception as e:
            print("締切取得失敗", rid, e, flush=True); continue
        if hm and not (ent["class"] == "L1").any():          # ガールズは対象外
            out.append((deadline_ts(date, hm), venue, rid))
    return sorted(out)


def judge(date, venue, rid, model, rules, bank, dl=None, inv_rules=None):
    html = kdreams.fetch(f"/{venue}/racedetail/{rid}/", refresh=True)
    ent, od, _, info, _, his = kdreams.parse_race(html, rid, extra=True)
    t = now()
    day = LIVE / date
    day.mkdir(parents=True, exist_ok=True)
    snap = od.assign(snapshot="live", taken_at=t.strftime("%H:%M:%S"))
    snap.to_csv(day / "odds_live.csv.gz", mode="a", index=False, compression="gzip",
                header=not (day / "odds_live.csv.gz").exists())
    if model is None or od.empty:
        return [], ent, []
    con = connect(":memory:")
    races = pd.DataFrame([dict(race_id=rid, date=date, venue=venue, race_no=int(rid[-4:]),
                               n_riders=len(ent), bank_len=kdreams.BANK.get(venue, 400),
                               wind=info["wind"])])
    store(con, dict(races=races, entries=ent, odds_trifecta=od))
    X = build_tensors(con, race_ids=[rid], need_result=False)
    if not X["T"].race_ids:
        return [], ent, []
    P = model.predict(X["T"], X["Q"])[0]
    cars = X["T"].cars[0]
    bets = []
    for kind, rl in (("yoso", rules), ("invest", inv_rules)):
        if rl is None:
            continue
        for n, st, p, o, ev in pick_bets(P, X["O"][0], bank, rl):
            i, j, k = COMBOS[n]
            bets.append(dict(date=date, race_id=rid, venue=venue, race_no=int(rid[-4:]),
                             deadline=dl.strftime("%H:%M") if dl is not None else "",
                             combo=f"{cars[i]}-{cars[j]}-{cars[k]}", stake=st, odds=o,
                             p=round(p, 5), ev=round(ev, 3), taken_at=t.strftime("%H:%M:%S"),
                             kind=kind))
    try:                                         # 一言解説（買い目の有無にかかわらず記録）
        from .insights import race_notes
        notes, line_probs = race_notes(P, X["T"], ent, his, od)
    except Exception as e:
        print("解説作成失敗", rid, e, flush=True); notes, line_probs = [], []
    with open(day / "notes.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(race_id=rid, venue=venue, race_no=int(rid[-4:]),
                                deadline=dl.strftime("%H:%M") if dl is not None else "",
                                notes=notes, lines=line_probs,
                                n_bets=sum(b["kind"] == "yoso" for b in bets),
                                n_invest=sum(b["kind"] == "invest" for b in bets)),
                           ensure_ascii=False) + "\n")
    if bets:
        pd.DataFrame(bets).to_csv(day / "bets.csv", mode="a", index=False,
                                  header=not (day / "bets.csv").exists())
    return bets, ent, notes


def _fmt_block(bets, names):
    from .strategy import summary
    sm = summary([(0, b["stake"], b["p"], b["odds"], b["ev"]) for b in bets])
    L = [f"**{len(bets)}点 {sm['total']:,}円**／合成オッズ {sm['synth']:.1f}倍／的中率 {sm['hit_p']:.0%}／期待値 {sm['race_ev']:.2f}", ""]
    by_first = {}
    for b in bets:
        by_first.setdefault(int(b["combo"].split("-")[0]), []).append(b)
    for f, bs in sorted(by_first.items(), key=lambda x: -sum(b["p"] for b in x[1])):
        L += [f"1着 **{f} {names.get(f, '')}**", "| 買い目 | 金額 | オッズ |", "|---|---:|---:|"]
        L += [f"| {b['combo']} | {b['stake']:,}円 | {b['odds']:.1f} |" for b in sorted(bs, key=lambda b: b["odds"])]
        L.append("")
    return L


def fmt(venue, rid, dl, bets, ent, mode, notes=None):
    names = dict(zip(ent["car"], ent["name"]))
    yoso = [b for b in bets if b.get("kind", "yoso") == "yoso"]
    inv = [b for b in bets if b.get("kind") == "invest"]
    head = f"### {kdreams.VENUE_JA.get(venue, venue)} {int(rid[-4:])}R（締切 {dl:%H:%M}）"
    if inv:
        head += " ★投資判定あり"
    L = [head + ("【紙上運用】" if mode == "paper" else ""), ""]
    if yoso:
        L += ["#### 本命予想（的中重視）"] + _fmt_block(yoso, names)
    if inv:
        L += ["#### ★投資判定（期待値で勝てる目）"] + _fmt_block(inv, names)
    if notes:
        L += ["**ひとこと**"] + [f"- {n}" for n in notes] + [""]
    L.append(f"判定時刻 {bets[0]['taken_at']}")
    return "\n".join(L)


def push_now(msg):
    """買い目を出すたびにリポジトリへ保存（ページへの反映を早くするため）"""
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    import subprocess
    sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
    sh('git config user.name "keirin-bot"; git config user.email "bot@users.noreply.github.com"')
    sh("git add data/live site")
    if sh(f'git commit -qm "{msg}"').returncode != 0:
        return
    for _ in range(3):
        if sh("git pull -q --rebase -X theirs && git push -q").returncode == 0:
            return
        time.sleep(5)
    print("保存失敗（実行の最後にまとめて保存します）", flush=True)


def run(max_minutes=340):
    t0 = time.time()
    date = now().strftime("%Y-%m-%d")
    if now().hour < 4:                         # 日付をまたいだミッドナイト
        date = (now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cfg, rules = load_config()
    inv_rules = load_invest_rules(cfg)
    mode, bank = cfg.get("mode", "paper"), cfg.get("bankroll", 100_000)
    try:
        model = TrifectaModel.from_json("models/latest.json")
    except Exception as e:
        print("モデルなし（オッズの記録のみ行います）:", e); model = None
    kdreams.SLEEP_SEC = 1.0
    full = schedule(date)
    sched = [x for x in full if x[0] > now() + pd.Timedelta(minutes=LEAD_MIN)]
    print(f"{date}: 判定対象 {len(sched)} レース", flush=True)
    try:                                   # 今日の締切一覧をページへ
        day = LIVE / date; day.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([dict(venue=v, race_no=int(r[-4:]), race_id=r, deadline=d.strftime("%H:%M"))
                      for d, v, r in full]).to_csv(day / "schedule.csv", index=False)
        from .site import export_live
        export_live(date)
        push_now("live 締切一覧")
    except Exception as e:
        print("締切一覧の保存失敗:", e, flush=True)
    note = Notifier(date)
    done = set()
    while sched and time.time() - t0 < max_minutes * 60:
        t = now()
        for dl, venue, rid in sched:
            left = (dl - t).total_seconds() / 60
            if rid in done or left > LEAD_MAX:
                continue
            done.add(rid)
            if left < LEAD_MIN:
                continue
            try:
                bets, ent, notes = judge(date, venue, rid, model, rules, bank, dl, inv_rules)
            except Exception as e:
                print("判定失敗", rid, e, flush=True); continue
            print(f"{t:%H:%M} {venue} {int(rid[-4:])}R 締切{dl:%H:%M} 買い目{len(bets)}点", flush=True)
            if bets:
                note.post(fmt(venue, rid, dl, bets, ent, mode, notes))
            try:                           # 買い・見送りにかかわらずページを更新
                from .site import export_live
                export_live(date)
            except Exception as e:
                print("サイト用データ更新失敗:", e, flush=True)
            push_now(f"live {venue} {int(rid[-4:])}R {'買い' if bets else '見送り'}")
        sched = [x for x in sched if x[2] not in done]
        time.sleep(30)
    print("終了", flush=True)


# ---------- 紙上運用の成績 ----------
def paper_results():
    from . import store as st
    bets = []
    for f in sorted(LIVE.glob("*/bets.csv")):
        bets.append(pd.read_csv(f, dtype={"race_id": str}))
    if not bets:
        return None
    b = pd.concat(bets, ignore_index=True)
    if "kind" not in b:
        b["kind"] = np.nan
    b["kind"] = b["kind"].fillna("invest")       # 区分導入前の買い目は期待値型
    res = []
    for d in b["date"].unique():
        f = st.day_dir(d) / "results.csv.gz"
        if f.exists():
            res.append(pd.read_csv(f, dtype={"race_id": str}))
    if not res:
        return b.assign(hit=np.nan, payout=np.nan)
    r = pd.concat(res)
    r["win"] = r["first"].astype(str) + "-" + r["second"].astype(str) + "-" + r["third"].astype(str)
    b = b.merge(r[["race_id", "win", "tri_payout"]], on="race_id", how="left")
    b["hit"] = b["combo"] == b["win"]
    b["payout"] = np.where(b["hit"], b["stake"] / 100 * b["tri_payout"], 0)
    b.loc[b["win"].isna(), ["hit", "payout"]] = np.nan
    return b
