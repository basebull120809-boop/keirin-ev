"""WINTICKET 収集アダプタ ── 「データ分析」用（調査段階）

robots.txt（2026/11 確認）: Disallow: /keirin/cups/ のみ。このパスにはアクセスしない。
WINTICKET はページを JavaScript で組み立てるため、ページに埋め込まれたデータ（__NEXT_DATA__ 等）と
ページが呼ぶ API を調べてから解析を書く。
方針は kdreams.py と同じ：取得したものは保存し、解析は保存済みから行う。アクセス間隔は SLEEP_SEC 秒以上。
利用規約を確認のうえ、個人利用の範囲で使うこと。
"""
import os
import re
import time
import hashlib
from pathlib import Path

BASE = "https://www.winticket.jp"
SLEEP_SEC = float(os.environ.get("KEIRIN_SLEEP", "2.0"))
CACHE = Path(os.environ.get("WINTICKET_CACHE", "data/html_wt"))
UA = "Mozilla/5.0 (personal research; keirin_ev)"
DISALLOW = ("/keirin/cups/",)
_session = None


def fetch(path, refresh=False):
    import requests
    global _session
    url = path if path.startswith("http") else BASE + path
    if url.startswith(BASE) and any(url[len(BASE):].startswith(d) for d in DISALLOW):
        raise RuntimeError(f"robots.txt で禁止: {url}")
    key = hashlib.md5(url.encode()).hexdigest()
    f = CACHE / f"{key}.html"
    if f.exists() and not refresh:
        return f.read_text(encoding="utf-8")
    if _session is None:
        _session = requests.Session()
    time.sleep(SLEEP_SEC)
    r = _session.get(url, headers={"User-Agent": UA, "Referer": BASE + "/"}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {url}")
    r.encoding = r.apparent_encoding or "utf-8"
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(r.text, encoding="utf-8")
    return r.text


RACE_LINK = re.compile(r'/keirin/[a-z]+/racecard/\d+/\d+/\d+')


def probe(out="debug/winticket", log=print):
    """出走表一覧 → 1レースのページ（タブ含む）→ ページが読む JS から API の場所を探して保存する"""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)

    def save(name, path):
        try:
            h = fetch(path, refresh=True)
        except Exception as e:
            log(f"  {name}: 失敗 {e}"); return None
        ext = "json" if h.lstrip()[:1] in "{[" else "html"
        (out / f"{name}.{ext}").write_text(h, encoding="utf-8")
        nd = "__NEXT_DATA__" in h
        log(f"  {name}: {len(h)//1024}KB  NEXT_DATA={nd}  %の数={len(re.findall(r'[0-9.]+%', h))}")
        return h

    save("robots_api", "https://api.winticket.jp/robots.txt")
    lst = save("racecard_top", "/keirin/racecard") or ""
    links = sorted(set(RACE_LINK.findall(lst)))
    log(f"レースへのリンク {len(links)}件: {' '.join(links[:5])}")
    (out / "links.txt").write_text("\n".join(sorted(set(re.findall(r'href="(/keirin/[^"]+)"', lst)))), encoding="utf-8")
    if not links:
        return
    race = links[len(links) // 2]
    h = save("race", race) or ""
    tabs = sorted(set(re.findall(r'href="(' + re.escape(race) + r'[^"]*)"', h)))
    log(f"{race}  タブ: {' '.join(tabs)}")
    for i, t in enumerate(tabs[:10]):
        save(f"race_tab{i}", t)
    # JS からAPIのホスト・パスを洗い出す
    js = sorted(set(re.findall(r'src="([^"]+\.js)"', h)))
    hits = set()
    for i, j in enumerate(js[:40]):
        u = j if j.startswith("http") else BASE + j
        try:
            s = fetch(u, refresh=True)
        except Exception:
            continue
        for m in re.findall(r'["\'`](https://[a-z0-9.\-]*winticket[a-z0-9.\-]*/[^"\'`\s]{0,120})', s):
            hits.add(m)
        for m in re.findall(r'["\'`](/v\d/keirin/[^"\'`\s]{0,120})', s):
            hits.add(m)
    (out / "api_refs.txt").write_text("\n".join(sorted(hits)), encoding="utf-8")
    log(f"JS {len(js)}本から API候補 {len(hits)}件: {' '.join(sorted(hits)[:15])}")
