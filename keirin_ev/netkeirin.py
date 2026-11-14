"""netkeirin（ネットケイリン）収集アダプタ ── 「データ分析」タブ用

URL規則（2026/11 時点で確認）
  出走表     : /race/entry/?race_id=YYYYMMDD + 場コード2桁 + R番号2桁
  データ分析 : /race/data/?race_id=...   選手ごとの条件別成績
               （レース種目別・周長・見なし直線・競輪場別・位置別・時間帯別・日程別・
                 グレード別・車番別・同県選手同乗時・同一ライン［位置別勝率］）
  直近成績   : /race/entry/results.html?race_id=...
  対戦表     : /race/match_list/?race_id=...
  場の開催   : /race/course/?jyo_cd=NN

方針は kdreams.py と同じ：取得したHTMLは保存し、解析は保存済みHTMLから行う。
アクセス間隔は SLEEP_SEC 秒以上。robots.txt は全許可（2026/11 確認）。
利用規約を確認のうえ、個人利用の範囲で使うこと。
"""
import os
import re
import time
import hashlib
from pathlib import Path

BASE = "https://keirin.netkeiba.com"
SLEEP_SEC = float(os.environ.get("KEIRIN_SLEEP", "2.0"))
CACHE = Path(os.environ.get("NETKEIRIN_CACHE", "data/html_nk"))
UA = "Mozilla/5.0 (personal research; keirin_ev)"
_session = None


def fetch(path, refresh=False, params=None):
    import requests
    global _session
    url = path if path.startswith("http") else BASE + path
    key = hashlib.md5((url + repr(sorted((params or {}).items()))).encode()).hexdigest()
    f = CACHE / f"{key}.html"
    if f.exists() and not refresh:
        return f.read_text(encoding="utf-8")
    if _session is None:
        _session = requests.Session()
    time.sleep(SLEEP_SEC)
    r = _session.get(url, params=params, headers={"User-Agent": UA, "Referer": BASE + "/"}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.url}")
    r.encoding = r.apparent_encoding or "utf-8"
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(r.text, encoding="utf-8")
    (CACHE / "index.tsv").open("a", encoding="utf-8").write(f"{key}\t{r.url}\n")
    return r.text


# 調査用に保存するレース（検索で見つかった実在のレース）
PROBE_RACES = ["202611118412",   # 武雄 2026/11/11 12R 決勝
               "202611074812"]   # 四日市 2026/11/07 12R 準決勝


# データ分析のカテゴリ（ページの左メニュー data_menu_*）と、type=0（まとめ）以外に見ておく区分
CATEGORIES = ["pickup", "class", "perimeter", "deemed", "jyo", "position", "time", "schedule",
              "grade", "gradeclass", "syaban", "samepref", "line", "special"]
EXTRA_TYPES = {"position": ("1", "2"), "perimeter": ("1",)}


def probe(date=None, out="debug/netkeirin", log=print):
    """データ分析ページの作りを調べる。
    レースのタブ一式・場の開催ページ・データ分析ページが読み込む JS / API を保存し、一覧を出す。
    date（YYYY-MM-DD）を渡すと、その日のレース一覧ページらしきものも試す"""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)

    def save(name, path, **kw):
        try:
            h = fetch(path, refresh=True, **kw)
        except Exception as e:
            log(f"  {name}: 失敗 {e}"); return None
        (out / f"{name}.html").write_text(h, encoding="utf-8")
        pct = len(re.findall(r"\d+(?:\.\d+)?%", h))
        log(f"  {name}: {len(h)//1024}KB  %の数={pct}")
        return h

    for rid in PROBE_RACES:
        log(f"race_id={rid}")
        h = save(f"{rid}_data", f"/race/data/?race_id={rid}")
        save(f"{rid}_entry", f"/race/entry/?race_id={rid}")
        save(f"{rid}_results", f"/race/entry/results.html?race_id={rid}")
        save(f"{rid}_match", f"/race/match_list/?race_id={rid}")
        if not h:
            continue
        # データ分析ページが読み込むスクリプト・API を洗い出す
        refs = sorted(set(re.findall(r'(?:src|href|url|data-url)\s*[=:]\s*["\']([^"\']+)["\']', h)))
        js = [r for r in refs if r.endswith(".js") or ".js?" in r]
        api = sorted(set(re.findall(r'["\']([^"\']*(?:api|ajax|json|\.php)[^"\']*)["\']', h, re.I)))
        (out / f"{rid}_refs.txt").write_text("\n".join(refs) + "\n\n# api\n" + "\n".join(api), encoding="utf-8")
        log(f"  JS {len(js)}件 / API候補 {len(api)}件: {' '.join(api[:10])}")
        for i, j in enumerate([x for x in js if "netkeiba" in x or x.startswith("/")][:8]):
            u = j if j.startswith("http") else (("https:" + j) if j.startswith("//") else BASE + j)
            save(f"{rid}_js{i}", u)
        # データ分析の中身（ページ表示後に読み込まれる部分）
        top = save(f"{rid}_ajax_top", f"/race/data/ajax_race_data_top.html?race_id={rid}")
        for cat in CATEGORIES:
            for typ in ("0",) + EXTRA_TYPES.get(cat, ()):
                save(f"{rid}_ajax_{cat}_{typ}", "/race/data/ajax_race_data.html",
                     params=dict(pid="ajax_race_data", input="UTF-8", output="html", race_id=rid,
                                 category=cat, type=typ, type2="2" if cat == "gradeclass" else "",
                                 select_matome_flag="true" if typ == "0" else "false"))
    save("course_84", "/race/course/?jyo_cd=84")
    save("race_top", "/race/")
    if date:
        ymd = date.replace("-", "")
        save(f"race_list_{ymd}", "/race/race_list/", params={"kaisai_date": ymd})


# ---------------- 解析 ----------------
def ajax_params(rid, cat, typ="0"):
    return dict(pid="ajax_race_data", input="UTF-8", output="html", race_id=rid, category=cat,
                type=typ, type2="2" if cat == "gradeclass" else "",
                select_matome_flag="true" if typ == "0" else "false")


def fetch_data(rid, cat, typ="0", refresh=False):
    return fetch("/race/data/ajax_race_data.html", refresh=refresh, params=ajax_params(rid, cat, typ))


def parse_data(html, rid=None, cat=None):
    """データ分析（ajax_race_data.html）→ 1行=選手×項目
    列: race_id, category, car, senshu, name, item, first, second, third, out, n, main（今回の条件の行なら True）"""
    import pandas as pd
    from bs4 import BeautifulSoup
    sp = BeautifulSoup(html, "lxml")
    tbs = [t for t in sp.find_all("table") if "Data01_Table" in (t.get("class") or [])
           and "Data01_Table2" not in t.get("class")]
    # 詳細の表は車番がすべて「1」になっているため、選手一覧の表（Data01_Table2）から車番を引く
    cars = {}
    for t in sp.find_all("table", class_="Data01_Table2"):
        for tr in t.find_all("tr", class_="PlayerList"):
            n, a = tr.find(class_="Num"), tr.find("a", class_="Player01")
            m = re.match(r"names_(\d+)_", a.get("id", "")) if a else None
            if n and m and n.get_text(strip=True).isdigit():
                cars[m.group(1)] = int(n.get_text(strip=True))
    rows, car, senshu, name = [], None, None, None
    for t in tbs[:1]:
        for tr in t.find_all("tr"):
            cls = tr.get("class") or []
            if "PlayerList" in cls:
                n = tr.find(class_="Num")
                car = int(n.get_text(strip=True)) if n and n.get_text(strip=True).isdigit() else None
                a = tr.find("a", class_="Player01")
                m = re.match(r"names_(\d+)_", a.get("id", "")) if a else None
                senshu, name = (m.group(1) if m else None), (a.get_text(strip=True) if a else None)
                car = cars.get(senshu, car)
                continue
            title = tr.find(class_="Data_Title")
            if not title or car is None:
                continue
            nums = [c.get_text(strip=True) for c in tr.find_all("td") if c is not title]
            v = [int(x) if x.isdigit() else None for x in nums[:5]]
            if len(v) < 5:
                continue
            rows.append(dict(race_id=rid, category=cat, car=car, senshu=senshu, name=name,
                             item=title.get_text(" ", strip=True), first=v[0], second=v[1], third=v[2],
                             out=v[3], n=v[4], main="Main_Data" in cls))
    return pd.DataFrame(rows)


def check_asof(kaisai="2026110984", days=("20261109", "20261110", "20261111"), out="debug/netkeirin", log=print):
    """過去レースのデータ分析が「そのレース時点」の集計かを確かめる。
    同じ開催（3日間）の全レースの「レース種目別・まとめ」を取り、同じ選手の出走回数が日を追って増えるかを見る。
    増えていれば時点集計（学習に使える）、全日同じなら今の集計（学習に使うとカンニングになる）"""
    import pandas as pd
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    jyo = kaisai[-2:]
    D = []
    for d in days:
        for r in range(1, 13):
            rid = f"{d}{jyo}{r:02d}"
            try:
                df = parse_data(fetch_data(rid, "class", refresh=True), rid, "class")
            except Exception as e:
                log(f"  {rid}: 失敗 {e}"); continue
            if df.empty:
                continue
            g = df.groupby(["senshu", "name"], as_index=False)["n"].sum()
            g["date"], g["race_id"] = d, rid
            D.append(g)
    if not D:
        log("データなし"); return
    A = pd.concat(D)
    A.to_csv(out / "asof_check.csv", index=False)
    piv = A.pivot_table(index=["senshu", "name"], columns="date", values="n", aggfunc="first")
    piv = piv.dropna()
    log(f"3日とも出走した選手 {len(piv)}人")
    log(piv.head(15).to_string())
    if len(piv):
        inc = (piv.diff(axis=1).iloc[:, 1:] > 0).all(axis=1).mean()
        same = (piv.nunique(axis=1) == 1).mean()
        log(f"日ごとに出走回数が増えた選手 {inc:.0%} ／ 3日とも同じ {same:.0%}")
        log("→ 時点集計（学習に使える）" if inc >= 0.8 else
            "→ 今の集計の可能性が高い（学習に使うとカンニング）" if same >= 0.8 else "→ 判断できず（要確認）")
