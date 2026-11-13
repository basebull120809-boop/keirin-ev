"""2車単・3連複などのオッズと払戻がページのどこにあるかを調べて debug/ に書き出す（一時的な調査用）"""
import re, sys, json
from pathlib import Path
sys.path.insert(0, ".")
from keirin_ev import kdreams as kd

out = Path("debug"); out.mkdir(exist_ok=True)
L = []
ids = []
for d in ["2026-11-12", "2026-11-11"]:
    try:
        ids += [(v, r) for v, r in kd.race_ids_of_day(d)][:2]
    except Exception as e:
        L.append(f"{d}: {e}")
for venue, rid in ids[:2]:
    base = f"/{venue}/racedetail/{rid}/"
    for q in ["", "?pageType=odds&kakeshikiType=3rentan", "?pageType=odds&kakeshikiType=2rentan",
              "?pageType=odds&kakeshikiType=3renpuku", "?pageType=odds&kakeshikiType=2renpuku",
              "?pageType=odds&kakeshikiType=wide"]:
        try:
            html = kd.fetch(base + q)
        except Exception as e:
            L.append(f"{rid}{q}: ERR {e}"); continue
        s = kd.soup(html)
        ids_ = sorted({t.get("id") for t in s.find_all(id=re.compile("ODDS|odds"))})
        L.append(f"== {venue} {rid} {q or '(detail)'} len={len(html)}")
        L.append("  ids: " + ", ".join(ids_))
        for i in ids_:
            box = s.find(id=i)
            tabs = box.find_all("table")
            L.append(f"  [{i}] tables={len(tabs)} classes={sorted({' '.join(t.get('class') or []) for t in tabs})}")
            if tabs:
                L.append("    first table text: " + re.sub(r"\s+", " ", tabs[0].get_text(" "))[:600])
        if not q:
            rt = s.find("table", class_="refund_table")
            if rt is not None:
                L.append("  refund: " + re.sub(r"\s+", " ", rt.get_text(" "))[:1200])
                L.append("  refund html: " + str(rt)[:2500])
        if q.endswith("2rentan") or q.endswith("3renpuku"):
            (out / f"sample_{q.split('=')[-1]}.html").write_text(html[:400000], encoding="utf-8")
(out / "kakeshiki_probe.txt").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
