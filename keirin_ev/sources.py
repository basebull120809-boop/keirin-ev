"""ほかの競輪サイトの調査（どのサイトから何が取れるか）

各サイトについて
  1. robots.txt を取り、Disallow を記録する（禁止されたパスにはアクセスしない）
  2. トップ・出走表らしいページを1〜2枚取り、
     ステータス・大きさ・JavaScriptで組み立てるページか・競輪の用語がどれだけ出るか を記録する
結果は debug/sources/ に保存する（HTMLは先頭だけ）。アクセス間隔は SLEEP_SEC 秒以上。
"""
import os
import re
import time
import json
from pathlib import Path
from urllib.parse import urlparse

SLEEP_SEC = float(os.environ.get("KEIRIN_SLEEP", "2.0"))
UA = "Mozilla/5.0 (personal research; keirin_ev)"

# サイト名, 調べるページ
SITES = [
    ("KEIRIN.JP（公式）", ["https://keirin.jp/", "https://keirin.jp/pc/top"]),
    ("オッズパーク", ["https://www.oddspark.com/keirin/", "https://www.oddspark.com/keirin/RaceList.do"]),
    ("チャリロト", ["https://www.chariloto.com/", "https://www.chariloto.com/keirin"]),
    ("Gamboo", ["https://www.gamboo.jp/", "https://www.gamboo.jp/keirin/"]),
    ("SPAIA競輪", ["https://spaia-keirin.jp/", "https://spaia.jp/keirin/"]),
    ("JKA", ["https://www.jka-cycle.jp/"]),
    ("競輪ステーション", ["https://www.keirin-station.com/"]),
    ("WINTICKET", ["https://www.winticket.jp/keirin"]),
    ("Kドリームス（取得中）", ["https://keirin.kdreams.jp/"]),
    ("ネット競輪（取得中）", ["https://keirin.netkeirin.com/"]),
]

WORDS = ["出走表", "競走得点", "ギヤ", "並び", "決まり手", "上がり", "予想", "オッズ",
         "バック", "ホーム", "逃", "捲", "差", "勝率", "連対率", "選手コメント", "AI"]


def _get(sess, url):
    time.sleep(SLEEP_SEC)
    try:
        r = sess.get(url, headers={"User-Agent": UA}, timeout=30, allow_redirects=True)
        r.encoding = r.apparent_encoding or "utf-8"
        return r.status_code, r.url, r.text
    except Exception as e:
        return None, url, f"{type(e).__name__}: {e}"


def probe(out="debug/sources", log=print):
    import requests
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    summary = []
    for name, pages in SITES:
        host = "{0.scheme}://{0.netloc}".format(urlparse(pages[0]))
        st, _, txt = _get(sess, host + "/robots.txt")
        dis = re.findall(r"(?im)^\s*disallow:\s*(\S+)", txt) if st == 200 else []
        log(f"■ {name}  robots.txt: {st}  Disallow {len(dis)}件 {' '.join(dis[:12])}")
        site = dict(name=name, robots=st, disallow=dis[:50], pages=[])
        for u in pages:
            path = urlparse(u).path or "/"
            # どの User-agent 向けかは区別せず、Disallow はすべて守る（安全側）
            if any(d and path.startswith(d.rstrip("*")) for d in dis):
                log(f"   {u}: robots.txt で禁止のため取得しない"); continue
            st, final, h = _get(sess, u)
            if st is None:
                log(f"   {u}: 失敗 {h}"); site["pages"].append(dict(url=u, status=None, err=h)); continue
            js = any(k in h for k in ("__NEXT_DATA__", "__NUXT__", 'id="root"', 'id="app"', "window.__INITIAL"))
            hits = {w: h.count(w) for w in WORDS if h.count(w)}
            links = sorted(set(re.findall(r'href="([^"#]+)"', h)))
            race_links = [l for l in links if re.search(r"(race|racecard|syusso|shusso|odds|result|yoso|predict)", l, re.I)]
            log(f"   {u} → {st} {final}  {len(h)//1024}KB  JS組立={js}  用語 {hits}")
            log(f"      レースらしいリンク {len(race_links)}件: {' '.join(race_links[:8])}")
            fn = re.sub(r"[^a-z0-9]+", "_", u.lower())[:80]
            (out / f"{fn}.html").write_text(h[:60000], encoding="utf-8")
            (out / f"{fn}.links.txt").write_text("\n".join(links), encoding="utf-8")
            site["pages"].append(dict(url=u, status=st, final=final, kb=len(h)//1024, js=js, words=hits,
                                      race_links=race_links[:40]))
        summary.append(site)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
