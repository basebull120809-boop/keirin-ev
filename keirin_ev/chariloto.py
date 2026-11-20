"""チャリロト「競輪予想」の取得（記者の印・周回予想（並び）・フォーカス・選手コメント）

ページ: https://www.chariloto.com/keirin/predictions/{日付}/{場コード}/detail
  場コードは JKA の2桁（race_id の先頭2桁と同じ。例: 22=前橋, 81=小倉）
  1ページにその場・その日の全レースが載る。
robots.txt（2026/11 確認）に Disallow なし。ページに「本レースデータを第三者に配布することを禁じます」とあるため、
取得したものは予想の特徴量として内部で使うだけにし、ページやダッシュボードにそのまま載せない。
アクセス間隔は SLEEP_SEC 秒以上。

保存先: data/daily/年/月/日付/chariloto.csv.gz
  列: race_id, car, mark, comment, line_order, line_no, focus
"""
import os
import re
import time
from pathlib import Path
import pandas as pd

BASE = "https://www.chariloto.com"
SLEEP_SEC = float(os.environ.get("KEIRIN_SLEEP", "2.0"))
UA = "Mozilla/5.0 (personal research; keirin_ev)"
_sess = None


def fetch(path):
    import requests
    global _sess
    if _sess is None:
        _sess = requests.Session()
    time.sleep(SLEEP_SEC)
    r = _sess.get(BASE + path, headers={"User-Agent": UA}, timeout=30)
    r.encoding = "utf-8"
    return r.status_code, r.text


def _text(s):
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"[\s　]+", " ", s).strip()


def parse(html):
    """予想ページ → 選手ごとの行（race_no, car, mark, comment）と、レースごとの周回予想・フォーカス"""
    rows, extra = [], {}
    # レースごとのブロック：「NR」の rowspan セルから次の「NR」まで
    blocks = re.split(r'<td rowspan="\d+" class="text-center">\s*<span class="text-bold">\s*(\d+)R\s*</span>\s*</td>', html)
    for i in range(1, len(blocks) - 1, 2):
        rno, body = int(blocks[i]), blocks[i + 1]
        tb = body.split("</tbody>")[0]
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", "<tr>" + tb, re.S):
            mark = re.search(r'<td height="36" class="bg-white text-center"><span>([^<]*)</span>', tr)
            car = re.search(r'class="bg-\d text-center text-bold">(\d)</td>', tr)
            tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
            if not car:
                continue
            rows.append(dict(race_no=rno, car=int(car.group(1)), mark=(mark.group(1).strip() if mark else ""),
                             comment=_text(tds[-1]) if tds else ""))
        # 周回予想：数字の並び。<span class="p10"> がラインの区切り
        m = re.search(r"周回予想</th>(.*?)</table>", body, re.S)
        order, line_no, ln = [], {}, 1
        if m:
            for tok in re.findall(r'square-slim bg-\d">(\d)</span>|(class="p10")', m.group(1)):
                if tok[0]:
                    c = int(tok[0]); order.append(c); line_no[c] = ln
                elif order:
                    ln += 1
        f = re.search(r"フォーカス</th>(.*?)</tr>", body, re.S)
        extra[rno] = dict(line_order=" ".join(map(str, order)), line_no=line_no,
                          focus=_text(f.group(1)) if f else "")
    df = pd.DataFrame(rows)
    if len(df):
        df["line_order"] = df.race_no.map(lambda r: extra.get(r, {}).get("line_order", ""))
        df["line_no"] = [extra.get(r, {}).get("line_no", {}).get(c) for r, c in zip(df.race_no, df.car)]
        df["focus"] = df.race_no.map(lambda r: extra.get(r, {}).get("focus", ""))
    return df


def fetch_day(date, races, log=print):
    """date の全場分を取る。races は race_id, date, race_no を持つその日のレース表"""
    races = races[races["date"] == date].copy()
    races["code"] = races["race_id"].astype(str).str[:2]
    out = []
    for code, g in races.groupby("code"):
        st, h = fetch(f"/keirin/predictions/{date}/{int(code)}/detail")
        if st != 200:
            log(f"  {date} 場{code}: HTTP {st}"); continue
        df = parse(h)
        if not len(df):
            log(f"  {date} 場{code}: 予想なし"); continue
        df = df.merge(g[["race_id", "race_no"]], on="race_no", how="inner")
        log(f"  {date} 場{code}: {df.race_id.nunique()}R {len(df)}行")
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def backfill(n_days=10, stop="2025-03-01", log=print):
    """まだ chariloto.csv.gz が無い日を新しい順に n_days 日分取る"""
    from .store import collected_dates, day_dir, read
    done = 0
    for d in sorted(collected_dates(), reverse=True):
        if d < stop or done >= n_days:
            break
        f = day_dir(d) / "chariloto.csv.gz"
        if f.exists():
            continue
        races = read(d, "races")
        if not len(races):
            continue
        df = fetch_day(d, races, log)
        df.to_csv(f, index=False, compression="gzip")       # 取れなかった日も空で保存（同じ日を何度も取りに行かない）
        done += 1
    log(f"{done}日分")


MARK = {"◎": 5, "○": 4, "▲": 3, "△": 2, "注": 1, "×": 1}


def read_all():
    from .store import collected_dates, day_dir
    fs = [day_dir(d) / "chariloto.csv.gz" for d in collected_dates()]
    L = []
    for f in fs:
        try:
            L.append(pd.read_csv(f, dtype={"race_id": str}))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass
    return pd.concat(L, ignore_index=True) if L else pd.DataFrame(columns=["race_id", "car", "mark"])
