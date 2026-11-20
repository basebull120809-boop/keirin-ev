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
    ("チャリロト", ["https://www.chariloto.com/keirin/predictions/list", "https://www.chariloto.com/keirin/results"]),
    ("Gamboo", ["https://gamboo.jp/", "https://gamboo.jp/keirin/"]),
    ("競輪ステーション", ["https://www.keirin-station.com/"]),
    ("ネット競輪", ["https://keirin.netkeiba.com/", "https://keirin.netkeiba.com/perfectanavi"]),
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


# ---------- 2段目：レースのページとタブを保存して中身を調べる ----------
DEEP = [
    # チャリロト：レースのページ・予想の詳細（日付・場コードは一覧から拾う）
    ("chariloto", "https://www.chariloto.com/keirin/predictions/list",
     r'href="(/keirin/(?:predictions/\d{4}-\d\d-\d\d/\d+/detail|\d{4}-\d\d-\d\d/\d+/\?race_num=\d+))"'),
    # Kドリームス：出走表のタブ（AI予想など）
    ("kdreams", "https://keirin.kdreams.jp/",
     r'href="(https://keirin\.kdreams\.jp/[a-z]+/racecard/\d+/)'),
    # ネット競輪：レース一覧 → レースのページ
    ("netkeiba", "https://keirin.netkeiba.com/race/",
     r'href="((?:https://keirin\.netkeiba\.com)?/race/[^"]*race_id=\d+[^"]*)"'),
]


def deep(out="debug/sources/deep", log=print, per_site=2):
    import requests
    from urllib.parse import urljoin
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()

    def save(tag, url):
        st, final, h = _get(sess, url)
        fn = tag + "_" + re.sub(r"[^a-z0-9]+", "_", url.lower().split("//", 1)[-1])[:90]
        if st == 200:
            (out / f"{fn}.html").write_text(h[:400000], encoding="utf-8")
        log(f"   {url} → {st} {len(h)//1024}KB  用語 { {w: h.count(w) for w in WORDS if h.count(w)} }")
        return h if st == 200 else ""

    for tag, top, pat in DEEP:
        log(f"■ {tag}: {top}")
        h = save(tag, top)
        cands = list(dict.fromkeys(re.findall(pat, h)))
        log(f"   候補 {len(cands)}件: {' '.join(cands[:6])}")
        for u in cands[:per_site]:
            u = urljoin(top, u.replace("&amp;", "&"))
            rh = save(tag, u)
            # レースのページ内のタブ（同じサイト内で、レースを表すIDを含むリンク）
            ids = re.findall(r"\d{8,}", u)
            tabs = set()
            for l in re.findall(r'href="([^"#]+)"', rh):
                l = urljoin(u, l.replace("&amp;", "&"))
                if urlparse(l).netloc == urlparse(u).netloc and l != u and (
                        any(i in l for i in ids) or ("race_num" in u and urlparse(u).path in l)):
                    tabs.add(l)
            log(f"   タブ {len(tabs)}件: {' '.join(sorted(tabs)[:25])}")
            for t in sorted(tabs)[:12]:
                save(tag, t)
