"""今日の残りレースを今のオッズで先に判定して data/preview/日付/ に保存する（当日運用とは別の記録。おすすめの決め方は当日運用と同じ）"""
import sys, time, json, subprocess, pathlib
sys.path.insert(0, ".")
import pandas as pd
from keirin_ev import live, kdreams
from keirin_ev.model import TrifectaModel

OUT = pathlib.Path("data/preview")
live.LIVE = OUT
kdreams.SLEEP_SEC = 2.0
date = live.now().strftime("%Y-%m-%d")
limit = float(sys.argv[1]) if len(sys.argv) > 1 else 300
cfg, rules = live.load_config()
inv = live.load_invest_rules(cfg)
model = TrifectaModel.from_json("models/latest.json")
sch = pd.read_csv(f"data/live/{date}/schedule.csv", dtype={"race_id": str})
t0 = time.time()


def save(msg):
    sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
    sh('git config user.name "keirin-bot"; git config user.email "bot@users.noreply.github.com"')
    sh("git add data/preview")
    if sh(f'git commit -qm "{msg}"').returncode == 0:
        for _ in range(3):
            if sh("git pull -q --rebase -X ours origin preview && git push -q origin HEAD:preview").returncode == 0:
                return
            time.sleep(5)


n = 0
for _, r in sch.iterrows():
    h, m = map(int, r.deadline.split(":"))
    dl = live.deadline_ts(date, (h, m))
    if dl < live.now() + pd.Timedelta(minutes=3):
        continue
    if (time.time() - t0) / 60 > limit:
        break
    try:
        live.judge(date, r.venue, r.race_id, model, rules, cfg.get("bankroll", 100000), dl=dl, inv_rules=inv)
        print("済", r.venue, r.race_no, r.deadline, flush=True)
    except Exception as e:
        print("失敗", r.venue, r.race_no, e, flush=True)
    n += 1
    if n % 8 == 0:
        save(f"preview {date} {n}R")
save(f"preview {date} 完了 {n}R")
