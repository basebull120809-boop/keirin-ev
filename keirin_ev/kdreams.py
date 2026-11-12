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
def race_ids_of_day(date):
    """date 'YYYY-MM-DD' → [(場slug, レースID16)]"""
    y, m, d = date.split("-")
    html = fetch(f"/kaisai/{y}/{m}/{d}/")
    kaisai = set(re.findall(r"/([a-z]+)/(?:raceresult|racecard)/(\d{14})/", html))
    out = []
    for venue, kid in sorted(kaisai):
        rhtml = fetch(f"/{venue}/raceresult/{kid}/")
        rids = sorted(set(re.findall(rf"/{venue}/racedetail/({kid[:12]}\d{{4}})/", rhtml)))
        out += [(venue, r) for r in rids]
    return out


# ---------------- 出走表 ----------------
def parse_entries(html, rid):
    s = soup(html)
    tbl = None
    for t in s.find_all("table"):
        h = _txt(t)
        if "競走得点" in h and "勝率" in h and "ギヤ" in h:
            tbl = t
            break
    if tbl is None:
        raise ValueError("出走表テーブルが見つかりません")
    rows = []
    for tr in tbl.find_all("tr"):
        cells = [_txt(td) for td in tr.find_all(["td", "th"])]
        # 「選手名 府県/年齢/期別」のセルを探す
        ci = next((i for i, c in enumerate(cells)
                   if re.search(r"/\s?\d+\s?/\s?\d+", c)), None)
        if ci is None:
            continue
        joined = " ".join(cells[ci:])
        cls = re.search(r"\b([SA][1-3]|SS|L1)\b", joined)
        style = re.search(r"\b(逃|両|追|自在)\b", joined)
        # 車番 = 選手名セルの直前にある1桁数字（その前が枠番）
        small = [c for c in cells[:ci] if re.fullmatch(r"[1-9]", c)]
        head = cells[ci].split("/")[0].split()
        name = "".join(head[:2]) if len(head) >= 3 else "".join(head)
        pref = "".join(head[2:]) if len(head) >= 3 else None
        after = re.sub(r"^.*?/\s?\d+\s?/\s?\d+", "", joined)
        f = re.findall(r"\d+\.\d+|\d+", after)
        # 期待並び: ギヤ 得点 S B 逃 捲 差 マ 1着 2着 3着 着外 勝率 2連対率 3連対率
        if not small or len(f) < 15:
            continue
        f = f[-15:]
        rows.append(dict(race_id=rid, car=int(small[-1]), name=name,
                         pref=pref, **{"class": cls.group(1) if cls else None},
                         style=style.group(1) if style else None,
                         gear=_num(f[0]), score=_num(f[1]), s_cnt=_num(f[2]),
                         back_cnt=_num(f[3]), win_rate=_num(f[12]),
                         ren_rate=_num(f[13]), san_rate=_num(f[14])))
    df = pd.DataFrame(rows).drop_duplicates("car")
    return df, parse_lines(s, df)


def parse_lines(s, entries):
    """並び予想 → line_id / line_pos / line_size
    DOM上のライン区切りが取れればそれを使い、取れなければ
    「自力型の選手からラインが始まる」という規則で区切る。"""
    node = s.find(string=re.compile("並び予想"))
    seq = []
    if node:
        box = node.find_parent()
        for _ in range(4):                          # 近い祖先から探す
            t = _txt(box)
            seq = re.findall(r"(?<!\d)([1-9])\s?(逃|先行|押え先|両|自在|まくり|捲|追込|追上|追)", t)
            if len(seq) >= len(entries) - 1:
                break
            box = box.find_parent() or box
    lines, cur = [], []
    for car, kind in seq:
        if cur and kind.startswith(JIRIKI):
            lines.append(cur); cur = []
        cur.append(int(car))
    if cur:
        lines.append(cur)
    rec = {}
    for li, ln in enumerate(lines):
        for p, c in enumerate(ln, 1):
            rec[c] = (li, p, len(ln))
    out = entries[["race_id", "car"]].copy()
    out["line_id"] = out["car"].map(lambda c: rec.get(c, (None,))[0])
    out["line_pos"] = out["car"].map(lambda c: rec.get(c, (None, None))[1])
    out["line_size"] = out["car"].map(lambda c: rec.get(c, (None, None, None))[2])
    return out


# ---------------- 結果 ----------------
def parse_result(html, rid):
    t = _txt(soup(html))
    m = re.search(r"3連単\s*([1-9])\s*[-ー－=]\s*([1-9])\s*[-ー－=]\s*([1-9])\s*([\d,]+)\s*円", t)
    if not m:
        return None
    return dict(race_id=rid, first=int(m.group(1)), second=int(m.group(2)),
                third=int(m.group(3)), tri_payout=int(m.group(4).replace(",", "")))


# ---------------- 3連単オッズ ----------------
def parse_odds(html, rid, n_riders):
    """1着ごとの格子表（行=2着, 列=3着）を読む"""
    s = soup(html)
    rows = []
    for t in s.find_all("table"):
        grid = [[_txt(td) for td in tr.find_all(["td", "th"])] for tr in t.find_all("tr")]
        grid = [g for g in grid if g]
        if len(grid) < 3:
            continue
        nums = sum(bool(re.fullmatch(r"\d+\.\d", c)) for g in grid for c in g)
        if nums < (n_riders - 2) * 2:
            continue
        # 1着: 表の直前にある「車番+名前」見出し
        head = t.find_previous(string=re.compile(r"^\s*[1-9]\s*\S+"))
        mh = re.match(r"\s*([1-9])", head or "")
        if not mh:
            continue
        first = int(mh.group(1))
        cols = [int(c) for c in grid[0] if re.fullmatch(r"[1-9]", c)]
        for g in grid[1:]:
            if not g or not re.fullmatch(r"[1-9]", g[0]):
                continue
            second = int(g[0])
            vals = [c for c in g[1:]]
            for third, v in zip(cols, vals):
                o = _num(v)
                if o and len({first, second, third}) == 3:
                    rows.append((rid, f"{first}-{second}-{third}", o, "final"))
    return pd.DataFrame(rows, columns=["race_id", "combo", "odds", "snapshot"]).drop_duplicates(["combo"])


# ---------------- 1日分 ----------------
class KdreamsCollector:
    name = "kdreams"

    def fetch_day(self, date, log=print):
        R, E, RES, OD = [], [], [], []
        for venue, rid in race_ids_of_day(date):
            path = f"/{venue}/racedetail/{rid}/"
            try:
                ent, ln = parse_entries(fetch(path), rid)
                ent = ent.merge(ln, on=["race_id", "car"], how="left")
                res = parse_result(fetch(path + "?pageType=KS_RACE_CARD_PAGE_TYPE_SHOW_RESULT"), rid)
                od = parse_odds(fetch(path + "?pageType=odds&kakeshikiType=3rentan"), rid, len(ent))
            except Exception as e:                     # 1レースの失敗で止めない
                log(f"  ! {rid} 解析失敗: {e}")
                continue
            n = len(ent)
            exp = n * (n - 1) * (n - 2)
            ok = "OK" if len(od) == exp and res else "要確認"
            log(f"  {venue} {int(rid[-4:])}R 選手{n} ライン{ent['line_id'].nunique()} "
                f"オッズ{len(od)}/{exp} 結果{'あり' if res else 'なし'} … {ok}")
            R.append(dict(race_id=rid, date=date, venue=venue, race_no=int(rid[-4:]),
                          n_riders=n, bank_len=BANK.get(venue, 400)))
            E.append(ent); OD.append(od)
            if res:
                RES.append(res)
        cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
        return dict(races=pd.DataFrame(R), entries=cat(E),
                    results=pd.DataFrame(RES), odds_trifecta=cat(OD))


def probe(dates, log=print):
    """何年前まで取れるかの確認。各日付の最初の1レースだけ取得して結果を表示"""
    for d in dates:
        try:
            ids = race_ids_of_day(d)
        except Exception as e:
            log(f"{d}: 開催一覧が取得できません ({e})"); continue
        if not ids:
            log(f"{d}: レースなし"); continue
        venue, rid = ids[0]
        path = f"/{venue}/racedetail/{rid}/"
        try:
            ent, ln = parse_entries(fetch(path), rid)
            od = parse_odds(fetch(path + "?pageType=odds&kakeshikiType=3rentan"), rid, len(ent))
            res = parse_result(fetch(path + "?pageType=KS_RACE_CARD_PAGE_TYPE_SHOW_RESULT"), rid)
            log(f"{d}: {len(ids)}レース / 例 {venue} {rid}: 選手{len(ent)} "
                f"得点{'あり' if ent['score'].notna().any() else 'なし'} "
                f"ライン{ln['line_id'].nunique()} オッズ{len(od)}点 結果{res}")
        except Exception as e:
            log(f"{d}: {len(ids)}レース / 解析失敗 {e}")
