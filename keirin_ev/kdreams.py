"""Kドリームス 収集アダプタ

URL規則（2026/11 時点で確認）
  開催一覧   : /kaisai/YYYY/MM/DD/
  結果一覧   : /{場}/raceresult/{開催ID14}/       開催ID14 = 場2桁+年月6桁+初日2桁+日目2桁+"00"
  出走表詳細 : /{場}/racedetail/{レースID16}/     レースID16 = 開催ID14の"00"をR番号4桁に
  結果       :  同 ?pageType=KS_RACE_CARD_PAGE_TYPE_SHOW_RESULT
  3連単オッズ:  同 ?pageType=odds&kakeshikiType=3rentan   (1着ごとに 2着×3着 の格子)

方針
  - 取得したHTMLはすべて data/html/ に保存し、解析は保存済みHTMLから行う
    （解析ロジックを直しても再アクセス不要。サイトへの負荷も最小）
  - アクセス間隔は SLEEP_SEC 秒以上
  - サイトの利用規約を確認のうえ、個人利用の範囲で使うこと
"""
import os
import re
import time
import hashlib
from pathlib import Path
import pandas as pd

BASE = "https://keirin.kdreams.jp"
SLEEP_SEC = float(os.environ.get("KEIRIN_SLEEP", "2.0"))
CACHE = Path(os.environ.get("KEIRIN_CACHE", "data/html"))
UA = "Mozilla/5.0 (personal research; keirin_ev)"

# バンク周長（不明な場は400扱い。必要に応じて修正）
BANK = {"matsudo": 333, "odawara": 333, "ito": 333, "nara": 333, "hofu": 333,
        "toyama": 333, "maebashi": 335, "utsunomiya": 500, "omiya": 500,
        "kochi": 500}

JIRIKI = ("逃", "先行", "押え先", "両", "自在", "まくり", "捲")   # 自力型
_session = None


# ---------------- 取得（キャッシュ付き） ----------------
def fetch(path, refresh=False):
    import requests
    global _session
    url = path if path.startswith("http") else BASE + path
    key = hashlib.md5(url.encode()).hexdigest()
    f = CACHE / f"{key}.html"
    if f.exists() and not refresh:
        return f.read_text(encoding="utf-8")
    if _session is None:
        _session = requests.Session()
    time.sleep(SLEEP_SEC)
    r = _session.get(url, headers={"User-Agent": UA}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {url}")
    r.encoding = r.apparent_encoding or "utf-8"
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(r.text, encoding="utf-8")
    (CACHE / "index.tsv").open("a", encoding="utf-8").write(f"{key}\t{url}\n")
    return r.text


def soup(html):
    from bs4 import BeautifulSoup
    return BeautifulSoup(html, "lxml")


def _txt(el):
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)) if el else ""


def _num(s):
    s = str(s).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


# ---------------- レース一覧 ----------------
def _kaisai_date(kid):
    """開催ID14 → その日の日付。場2+年月6+初日2+日目2+00"""
    start = pd.Timestamp(f"{kid[2:6]}-{kid[6:8]}-{kid[8:10]}")
    return start + pd.Timedelta(days=int(kid[10:12]) - 1)


def race_ids_of_day(date):
    """date 'YYYY-MM-DD' → [(場slug, レースID16)]  その日に行われたレースだけ"""
    y, m, d = date.split("-")
    html = fetch(f"/kaisai/{y}/{m}/{d}/")
    target = pd.Timestamp(date)
    kaisai = {(v, k) for v, k in re.findall(r"/([a-z]+)/(?:raceresult|racecard)/(\d{14})/", html)
              if _kaisai_date(k) == target}
    out = []
    for venue, kid in sorted(kaisai):
        rhtml = fetch(f"/{venue}/raceresult/{kid}/")
        rids = sorted(set(re.findall(rf"/{venue}/racedetail/({kid[:12]}\d{{4}})/", rhtml)))
        out += [(venue, r) for r in rids]
    return out


# ---------------- レースページ（出走表・ライン・オッズ・結果が1ページに入っている） ----------------
def parse_entries(s, rid):
    tbl = s.find("table", class_="racecard_table")
    if tbl is None:
        raise ValueError("出走表テーブルが見つかりません")
    rows = []
    for tr in tbl.find_all("tr"):
        num = tr.find("td", class_="num")
        rider = tr.find("td", class_="rider")
        if not num or not rider:
            continue
        tds = tr.find_all("td")
        i = tds.index(rider)
        rest = [td.get_text(strip=True) for td in tds[i + 1:]]
        parts = rider.get_text("|", strip=True).split("|")
        info = parts[1].split("/") if len(parts) > 1 else [None, None, None]
        # 級班 脚質 ギヤ 得点 S B 逃 捲 差 マ 1着 2着 3着 着外 勝率 2連対率 3連対率
        if len(rest) < 17:
            continue
        g = lambda k: _num(rest[k])
        rows.append(dict(race_id=rid, car=int(num.get_text(strip=True)),
                         name=parts[0].replace(" ", ""),
                         pref=(info[0] or "").replace("\u3000", "").replace(" ", ""),
                         age=_num(info[1]) if len(info) > 1 else None,
                         term=_num(info[2]) if len(info) > 2 else None,
                         **{"class": rest[0]}, style=rest[1], gear=g(2), score=g(3),
                         s_cnt=g(4), back_cnt=g(5), nige=g(6), makuri=g(7), sashi=g(8),
                         mark=g(9), n1=g(10), n2=g(11), n3=g(12), nout=g(13),
                         win_rate=g(14), ren_rate=g(15), san_rate=g(16)))
    if not rows:
        raise ValueError("出走表の行が読めません")
    return pd.DataFrame(rows).drop_duplicates("car")


def parse_lines(s, entries):
    """並び予想: span.icon_p.space がライン区切り"""
    box = s.find("div", class_="line_position")
    lines, cur = [], []
    if box:
        for sp in box.find_all("span", class_="icon_p", recursive=False):
            if "space" in (sp.get("class") or []):
                if cur:
                    lines.append(cur); cur = []
                continue
            inner = sp.find_all("span")
            if inner and re.fullmatch(r"[1-9]", inner[0].get_text(strip=True)):
                cur.append(int(inner[0].get_text(strip=True)))
        if cur:
            lines.append(cur)
    rec = {c: (li, p, len(ln)) for li, ln in enumerate(lines) for p, c in enumerate(ln, 1)}
    nxt = len(lines)
    out = entries[["race_id", "car"]].copy()
    ids, pos, size = [], [], []
    for c in out["car"]:
        if c in rec:
            l, p, z = rec[c]
        else:                       # 並びに無い選手は単騎扱い
            l, p, z = nxt, 1, 1; nxt += 1
        ids.append(l); pos.append(p); size.append(z)
    out["line_id"], out["line_pos"], out["line_size"] = ids, pos, size
    return out


def parse_odds(s, rid):
    box = s.find(id="JS_ODDSCONTENTS_3rentan")
    if box is None:
        return pd.DataFrame(columns=["race_id", "combo", "odds", "snapshot"])
    rows = []
    for t in box.find_all("table", class_="odds_table"):
        num = t.find("span", class_="number")
        if not num:
            continue
        first = int(num.get_text(strip=True))
        trs = t.find_all("tr")
        cols = [int(th.get_text(strip=True)) for th in trs[1].find_all("th")
                if re.fullmatch(r"[1-9]", th.get_text(strip=True))]
        for tr in trs[3:]:
            ths = tr.find_all("th")
            if not ths or not re.fullmatch(r"[1-9]", ths[0].get_text(strip=True)):
                continue
            third = int(ths[0].get_text(strip=True))      # 行=3着、列=2着
            for second, td in zip(cols, tr.find_all("td")):
                o = _num(td.get_text(strip=True))
                if o and len({first, second, third}) == 3:
                    rows.append((rid, f"{first}-{second}-{third}", o, "final"))
    return pd.DataFrame(rows, columns=["race_id", "combo", "odds", "snapshot"]).drop_duplicates("combo")


def parse_result(s, rid):
    rt = s.find("table", class_="refund_table")
    if rt is None:
        return None
    for dl in rt.find_all("dl"):
        dt, dd = dl.find("dt"), dl.find("dd")
        if not dt or not dd:
            continue
        m = re.fullmatch(r"([1-9])-([1-9])-([1-9])", dt.get_text(strip=True))
        if m:
            pay = re.search(r"([\d,]+)円", dd.get_text())
            pop = re.search(r"\((\d+)\)", dd.get_text())
            return dict(race_id=rid, first=int(m.group(1)), second=int(m.group(2)),
                        third=int(m.group(3)),
                        tri_payout=int(pay.group(1).replace(",", "")) if pay else None,
                        tri_pop=int(pop.group(1)) if pop else None)
    return None


def parse_race_info(s):
    t = _txt(s.find("p", class_="weather")) if s.find("p", class_="weather") else ""
    w = re.search(r"風速\s*([\d.]+)", t)
    wx = re.search(r"天候\s*(\S+?)/", t)
    return dict(wind=float(w.group(1)) if w else None, weather=wx.group(1) if wx else None)


def parse_race(html, rid):
    s = soup(html)
    ent = parse_entries(s, rid)
    ent = ent.merge(parse_lines(s, ent), on=["race_id", "car"], how="left")
    return ent, parse_odds(s, rid), parse_result(s, rid), parse_race_info(s)


# ---------------- 1日分 ----------------
def check(rid, ent, od, res):
    n = len(ent); exp = n * (n - 1) * (n - 2)
    msg = f"選手{n} ライン{ent['line_id'].nunique()} オッズ{len(od)}/{exp}"
    ok = len(od) == exp
    if res:
        combo = f"{res['first']}-{res['second']}-{res['third']}"
        hit = od.loc[od["combo"] == combo, "odds"]
        o = float(hit.iloc[0]) if len(hit) else None
        msg += f" 結果{combo} 払戻{res['tri_payout']}円 / 表のオッズ{o}"
        if o and res["tri_payout"]:
            ok &= abs(o * 100 - res["tri_payout"]) / res["tri_payout"] < 0.02
    else:
        msg += " 結果なし"; ok = False
    return ok, msg


class KdreamsCollector:
    name = "kdreams"

    def fetch_day(self, date, log=print):
        R, E, RES, OD = [], [], [], []
        n_ok = 0
        ids = race_ids_of_day(date)
        for venue, rid in ids:
            try:
                ent, od, res, info = parse_race(fetch(f"/{venue}/racedetail/{rid}/"), rid)
            except Exception as e:                     # 1レースの失敗で止めない
                log(f"  ! {venue} {rid} 解析失敗: {e}")
                continue
            ok, msg = check(rid, ent, od, res)
            n_ok += ok
            if not ok:
                log(f"  要確認 {venue} {int(rid[-4:])}R {msg}")
            R.append(dict(race_id=rid, date=date, venue=venue, race_no=int(rid[-4:]),
                          grade=ent["class"].iloc[0] if len(ent) else None,
                          n_riders=len(ent), bank_len=BANK.get(venue, 400), **info))
            E.append(ent); OD.append(od)
            if res:
                RES.append(res)
        log(f"  {len(ids)}レース中 {n_ok} レース完全一致（オッズ×100＝払戻）")
        cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
        return dict(races=pd.DataFrame(R), entries=cat(E),
                    results=pd.DataFrame(RES), odds_trifecta=cat(OD))


def probe(dates, log=print):
    """何年前まで取れるかの確認。各日付の先頭3レースを取得して検証"""
    for d in dates:
        try:
            ids = race_ids_of_day(d)
        except Exception as e:
            log(f"{d}: 開催一覧が取得できません ({e})"); continue
        log(f"{d}: {len(ids)}レース")
        for venue, rid in ids[:3]:
            try:
                ent, od, res, info = parse_race(fetch(f"/{venue}/racedetail/{rid}/"), rid)
                ok, msg = check(rid, ent, od, res)
                log(f"   {'OK ' if ok else 'NG '}{venue} {int(rid[-4:])}R {msg} 風{info['wind']}")
            except Exception as e:
                log(f"   NG {venue} {rid} 解析失敗 {e}")
