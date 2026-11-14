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
