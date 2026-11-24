"""当日運用：締切の数分前にオッズを取り直して期待値判定 → GitHubのIssueに通知

流れ
  1) 起動時に当日の全レースの締切時刻を取得
  2) 30秒ごとに見回り、締切まで LEAD_MAX〜LEAD_MIN 分のレースを判定
  3) 判定結果は data/live/日付/ に保存（オッズのスナップショット・買い目）
  4) 買い目があれば当日のIssueにコメント（GitHubアプリに通知が届く）

実際の購入はしない。通知を見て、買うかどうかは自分で決める。
"""
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

from . import kdreams
from .db import connect
from .collector import store
from .model import TrifectaModel, build_tensors, COMBOS

CONF_MIN = 0.40                                       # 買い目を出す自信の線（展開ベース3連複4点の見込み的中率）
AXIS_KINDS = ("yoso", "ex", "qn", "trio", "tenkai")   # 予想の軸にそろえる券種（invest・pool・hit4 はそのまま）
from .strategy import Rules, pick_bets

JST = "Asia/Tokyo"
LIVE = Path("data/live")
LEAD_MAX = float(os.environ.get("LEAD_MAX", "6"))   # 締切何分前から判定するか
LEAD_MIN = float(os.environ.get("LEAD_MIN", "1.5"))  # これより締切に近ければ見送り
RES_AFTER = float(os.environ.get("RES_AFTER", "8"))   # 締切何分後から結果を見に行くか
RES_GIVEUP = 90                                       # 締切からこの分数たっても出なければ諦める（翌朝のcollectで入る）
RES_EVERY = 3                                         # 結果待ちの再確認間隔（分）


def now():
    return pd.Timestamp.now(tz=JST)


def load_config():
    p = Path("config.json")
    cfg = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    rules = Rules(**{k: v for k, v in cfg.get("rules", {}).items()
                     if k in Rules.__dataclass_fields__})
    return cfg, rules


def load_invest_rules(cfg):
    """投資判定（期待値で勝てる目だけ）のルール。設定が無ければ使わない"""
    r = cfg.get("invest_rules")
    if not r:
        return None
    return Rules(**{k: v for k, v in r.items() if k in Rules.__dataclass_fields__})


# ---------- 通知（GitHub Issue） ----------
class Notifier:
    def __init__(self, date):
        self.repo = os.environ.get("GITHUB_REPOSITORY")
        self.token = os.environ.get("GITHUB_TOKEN")
        self.title = f"買い目 {date}"
        self.num = None

    def _api(self, method, path, **kw):
        import requests
        r = requests.request(method, f"https://api.github.com/repos/{self.repo}{path}",
                             headers={"Authorization": f"Bearer {self.token}",
                                      "Accept": "application/vnd.github+json"},
                             timeout=20, **kw)
        r.raise_for_status()
        return r.json()

    def post(self, text):
        print(text, flush=True)
        if not (self.repo and self.token):
            return
        try:
            if self.num is None:
                for it in self._api("GET", "/issues", params={"state": "open", "per_page": 100}):
                    if it["title"] == self.title:
                        self.num = it["number"]
                if self.num is None:
                    self.num = self._api("POST", "/issues", json={
                        "title": self.title,
                        "body": "締切前オッズで期待値判定した買い目をここに流します。"})["number"]
            self._api("POST", f"/issues/{self.num}/comments", json={"body": text})
        except Exception as e:
            print("通知失敗:", e, flush=True)


# ---------- 判定 ----------
def deadline_ts(date, hm):
    h, m = hm
    return pd.Timestamp(f"{date} 00:00", tz=JST) + pd.Timedelta(hours=h, minutes=m)


def schedule(date):
    """当日の [(締切, 場, レースID)]"""
    out = []
    for venue, rid in kdreams.race_ids_of_day(date):
        try:
            s = kdreams.soup(kdreams.fetch(f"/{venue}/racedetail/{rid}/", refresh=True))
            hm = kdreams.parse_deadline(s)
            ent = kdreams.parse_entries(s, rid)
        except Exception as e:
            print("締切取得失敗", rid, e, flush=True); continue
        if hm and not (ent["class"] == "L1").any():          # ガールズは対象外
            out.append((deadline_ts(date, hm), venue, rid))
    return sorted(out)


_CMP = []


def _compress_predictor():
    """圧縮モデルの当日予想（起動後の最初の判定で1回だけ過去データを読む。モデルが無ければ使わない）"""
    if not _CMP:
        try:
            try:
                import lightgbm  # noqa: F401
            except ImportError:                  # 起動後にコードだけ更新されて再起動した場合（requirements の入れ直しが無い）
                if os.environ.get("GITHUB_ACTIONS"):
                    import subprocess, sys
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm"], capture_output=True)
            from .compress import LivePredictor
            _CMP.append(LivePredictor())
        except Exception as e:
            print("圧縮モデルを使わない:", e, flush=True)
            _CMP.append(None)
    return _CMP[0]


def strategy():
    """config.json の strategy（"hit" = 的中重視の1方式だけ。それ以外・未設定なら従来の全方式）"""
    try:
        return json.loads(Path("config.json").read_text(encoding="utf-8")).get("strategy", "")
    except Exception:
        return ""


ODDS_CAP = 20.0     # どの買い方でも判定時オッズがこれを超える目は買わない（2026-11-24：実績11日・約5,000点で20倍超は的中率3%未満）


def _cap_odds(bets):
    """判定時オッズが ODDS_CAP 倍を超える目を外す（オッズが付いていない目はそのまま）"""
    out = []
    for b in bets:
        try:
            o = float(b.get("odds") or 0)
        except (TypeError, ValueError):
            o = 0
        if o <= ODDS_CAP:
            out.append(b)
    return out


HIT_SYNTH = 1.6      # 合成オッズの下限（判定時オッズ×0.93で見込む）
HIT_GO = 0.85        # 勝負（収支に入れる）にする見込み回収（当たる見込み×合成オッズ）の下限
HIT_HMIN = 0.50      # 勝負にする当たる見込みの下限（まず当てる：2026-11-24）
HIT_MINPTS = 2       # 1点買いはしない
HIT_DRIFT = 0.93     # 判定時→確定で当たり目の配当が下がる分
HIT_BUDGET = 1000    # 1レースの予算（どの目が当たっても払戻がそろう配分）


def _hit_plan(probs, odds, kind):
    """確率の高い目から足して、合成オッズ HIT_SYNTH 倍以上を保てるところまで（最大12候補を見る）"""
    cand = []
    for c, o in (odds or {}).items():
        try:
            o = float(o)
        except (TypeError, ValueError):
            continue
        from .pools import _key
        c = _key(c); p = probs.get(c, 0.0)
        if o > 0 and p > 0:
            cand.append((p, o, c))
    cand.sort(reverse=True)
    inv, sel = 0.0, []
    for p, o, c in cand[:12]:
        if 1 / (inv + 1 / (o * HIT_DRIFT)) < HIT_SYNTH:
            break
        inv += 1 / (o * HIT_DRIFT); sel.append((p, o, c))
    if len(sel) < HIT_MINPTS:                # 2点目で合成1.6倍を割る券種は使わない（1点買いはしない）
        return None
    H = sum(p for p, _, _ in sel); S = 1 / inv
    return dict(kind=kind, H=H, S=S, ev=H * S,
                bets=[dict(combo=c, stake=0, odds=o, p=round(p, 5), ev=round(p * o * HIT_DRIFT, 3)) for p, o, c in sel])


def _judge_hit(date, venue, rid, dl, t, day, html, ent, od, his, X, P_res):
    """ただ1つの方式（2026-11-24）：全レースで的中重視の予想を出す。

    残差モデルの確率で、2車複・3連複それぞれ「確率の高い目から、合成オッズ1.6倍以上（判定時オッズ×0.93で見込む）を
    保てるところまで」買う形を作る（最低2点。1点買いはしない）。当たる見込みが高い方を1つ出す。
    金額は どの目が当たっても払戻がそろう配分（1レース約1,000円）。
    当たる見込み50%以上かつ見込み回収0.85以上を「勝負」（収支に入れる）、それ以外は「見送り推奨」（予想は出す・収支外）。
    過去検証（2025-11〜2026-11・毎月学習し直し・確定オッズ）：全レースで的中45%・回収79%、
    勝負（1日約8R）で的中53%・回収89%（95%区間 86〜92%）。当てる方を優先したので回収率は100%に届かない
    """
    from .pools import _key
    from .scenario import guard_stakes
    base = dict(date=date, race_id=rid, venue=venue, race_no=int(rid[-4:]),
                deadline=dl.strftime("%H:%M") if dl is not None else "", taken_at=t.strftime("%H:%M:%S"))
    qno, tro = {}, {}
    try:
        sp = kdreams.soup(html)
        _oq, _ot = kdreams.parse_odds_quinella(sp, rid), kdreams.parse_odds_trio(sp, rid)
        for nm, df in (("quinella", _oq), ("trio", _ot)):    # 判定時オッズは検証用に保存を続ける
            if len(df):
                fo = day / f"odds_live_{nm}.csv.gz"
                df.assign(taken_at=t.strftime("%H:%M:%S")).to_csv(fo, mode="a", index=False, compression="gzip",
                                                                   header=not fo.exists())
        qno, tro = dict(_oq[["combo", "odds"]].values), dict(_ot[["combo", "odds"]].values)
    except Exception as e:
        print("2車複・3連複のオッズを読めず", rid, e, flush=True)
    cars = X["T"].cars[0]
    bets, reason, rec, why, mi, mi_why = [], "", None, "", None, ""
    if P_res is None:
        reason = "残差モデルが使えないため見送り"
    else:
        qp, tp = {}, {}
        for k, (i, j, l) in enumerate(COMBOS):
            p = float(P_res[k])
            if p <= 0 or not (cars[i] and cars[j] and cars[l]):
                continue
            a, b, c = int(cars[i]), int(cars[j]), int(cars[l])
            kq, kt = _key(f"{a}={b}"), _key(f"{a}={b}={c}")
            qp[kq] = qp.get(kq, 0.0) + p; tp[kt] = tp.get(kt, 0.0) + p
        plans = [x for x in (_hit_plan(qp, qno, "qn"), _hit_plan(tp, tro, "trio")) if x]
        if not plans:
            reason = "2点以上・合成オッズ1.6倍以上で組める目がないため見送り"
        else:
            pl = max(plans, key=lambda x: (x["H"], x["ev"]))   # まず当てる：当たる見込みの高い券種
            bl = pl["bets"]
            st = None
            while len(bl) >= HIT_MINPTS and st is None:              # 100円単位で払戻がそろう配分。無理なら確率の低い目から外す
                st = guard_stakes(bl, budget=max(HIT_BUDGET, 300 * len(bl)), drift=HIT_DRIFT, min_synth=HIT_SYNTH)
                if st is None:
                    bl = bl[:-1]
            if not st:
                reason = "100円単位・2点以上で合成1.6倍以上の配分にできないため見送り"
            else:
                lab = {"qn": "2車複", "trio": "3連複"}[pl["kind"]]
                txt = (f"{lab}{len(st)}点：当たる見込み{pl['H']:.0%}・合成{pl['S']:.1f}倍（確定見込み）"
                       f"・見込み回収{pl['ev']:.0%}")
                if pl["H"] >= HIT_HMIN and pl["ev"] >= HIT_GO:
                    bets = [dict(base, **b, kind=pl["kind"], rec=True) for b in st]
                    rec, why = pl["kind"], "勝負：" + txt
                else:
                    bets = [dict(base, **b, kind=pl["kind"] + "_m", rec=False) for b in st]
                    mi, mi_why = pl["kind"], txt
                    reason = (f"当たる見込みが{HIT_HMIN:.0%}に届かない（{pl['H']:.0%}）" if pl["H"] < HIT_HMIN
                              else f"見込み回収が{HIT_GO:.0%}に届かない（{pl['ev']:.0%}）")
        try:                                      # 回収重視の2車複（期待値1.0以上）も紙上で記録だけ続ける（表示・おすすめの収支には入れない）
            from . import evbuy
            _, b2 = evbuy.plan(P_res, cars, COMBOS, {}, qno)
            bets += [dict(base, **b, kind="evr2", rec=False) for b in b2]
        except Exception as e:
            print("回収重視の記録失敗", rid, e, flush=True)
    notes, line_probs = [], []
    try:
        from .insights import race_notes
        notes, line_probs = race_notes(P_res if P_res is not None else np.zeros(len(COMBOS)), X["T"], ent, his, od,
                                       date=date, venue=venue)
    except Exception as e:
        print("解説作成失敗", rid, e, flush=True)
    with open(day / "notes.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(race_id=rid, venue=venue, race_no=int(rid[-4:]),
                                deadline=dl.strftime("%H:%M") if dl is not None else "",
                                notes=notes, lines=line_probs,
                                names={int(c): str(nm) for c, nm in zip(ent["car"], ent["name"])},
                                n_bets=sum(1 for b in bets if b.get("rec")), n_invest=0, rec=rec, rec_reason=why,
                                mi=mi, mi_reason=mi_why, cand={}, reason=reason), ensure_ascii=False) + "\n")
    bets = _cap_odds(bets)
    if bets:
        f = day / "bets.csv"
        new = pd.DataFrame(bets)
        if f.exists():
            old = pd.read_csv(f, dtype={"race_id": str}, on_bad_lines="skip")
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(f, index=False)
    return [b for b in bets if b["kind"] != "evr2"], ent, notes


def judge(date, venue, rid, model, rules, bank, dl=None, inv_rules=None):
    html = kdreams.fetch(f"/{venue}/racedetail/{rid}/", refresh=True)
    ent, od, _, info, _, his = kdreams.parse_race(html, rid, extra=True)
    t = now()
    day = LIVE / date
    day.mkdir(parents=True, exist_ok=True)
    snap = od.assign(snapshot="live", taken_at=t.strftime("%H:%M:%S"))
    snap.to_csv(day / "odds_live.csv.gz", mode="a", index=False, compression="gzip",
                header=not (day / "odds_live.csv.gz").exists())
    if model is None or od.empty:
        with open(day / "notes.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(race_id=rid, venue=venue, race_no=int(rid[-4:]),
                                    deadline=dl.strftime("%H:%M") if dl is not None else "",
                                    notes=[], lines=[], n_bets=0, n_invest=0,
                                    reason="オッズが取得できなかったため" if od.empty else "予測モデルが未学習のため"),
                               ensure_ascii=False) + "\n")
        return [], ent, []
    con = connect(":memory:")
    races = pd.DataFrame([dict(race_id=rid, date=date, venue=venue, race_no=int(rid[-4:]),
                               n_riders=len(ent), bank_len=kdreams.BANK.get(venue, 400),
                               wind=info["wind"])])
    store(con, dict(races=races, entries=ent, odds_trifecta=od))
    nk = pd.DataFrame()
    try:                                         # ネット競輪のデータ分析（取れなければ使わずに判定）
        from . import netkeirin
        netkeirin.SLEEP_SEC = 0.7
        nk = netkeirin.fetch_race(rid, int(rid[-4:]), date)
        if not nk.empty:
            f = day / "nk_data.csv.gz"
            nk.drop(columns=["name"], errors="ignore").to_csv(f, mode="a", index=False, compression="gzip",
                                                              header=not f.exists())
    except Exception as e:
        print("データ分析の取得失敗", rid, e, flush=True)
    X = build_tensors(con, race_ids=[rid], need_result=False, history=his, nk=nk)
    if not X["T"].race_ids:
        return [], ent, []
    P = model.predict(X["T"], X["Q"])[0]
    use_res = False
    P_res = None
    try:                                         # 市場との差を直接学んだ残差モデルがあればそちらを使う（検証期間で旧モデルより良い）
        from .residual import ResidualModel
        P = ResidualModel.from_json("models/residual.json").predict(X["T"], X["Q"])[0]
        use_res = True
        P_res = P                                # 残差モデルの期待値買い（evbuy）用に取っておく
        try:                                     # 診断用：判定時の特徴量と残差モデルの確率を保存（過去検証との食い違いを調べるため）
            from .model import FEATURES
            T0 = X["T"]; mk = T0.mask[0]
            top = np.argsort(-P_res)[:30]
            with open(day / "diag.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(race_id=rid, taken_at=t.strftime("%H:%M:%S"),
                                        cars=[int(c) for c in T0.cars[0][mk]],
                                        X={fe: [round(float(v), 5) for v in T0.X[0][mk, i]] for i, fe in enumerate(FEATURES)},
                                        top=[[int(n), round(float(P_res[n]), 6), round(float(X["Q"][0][n]), 6)] for n in top]),
                                   ensure_ascii=False) + "\n")
        except Exception as e:
            print("診断の保存失敗", rid, e, flush=True)
    except FileNotFoundError:
        pass
    except Exception as e:
        print("残差モデルを使えず旧モデルで判定", rid, e, flush=True)
    if strategy() == "hit":                      # 2026-11-24 方式を1つに絞った：全レースで的中重視の予想（勝負は見込み回収0.95以上だけ）
        return _judge_hit(date, venue, rid, dl, t, day, html, ent, od, his, X, P_res)
    scen = None
    try:                                         # 展開シナリオ予想（全展開を想定＋オッズ）があれば最優先で使う
        from .scenario import ScenarioModel
        scen = ScenarioModel.from_json()
        P = scen.predict(X["T"], X["Q"])[0]
        use_res = True
    except FileNotFoundError:
        pass
    except Exception as e:
        scen = None
        print("展開シナリオ予想を使えず従来の予想で判定", rid, e, flush=True)
    cars = X["T"].cars[0]
    sim = None
    try:                                         # 予想前に1000回シミュレーション → あらゆる展開を混ぜた確率で組む
        from .simulate import simulate           # 1000回走らせて、展開の揺らぎ込みで「何回当たるか」を数える（買い目の選択に使う）
        sim = simulate(P, X["T"], 0, X["O"][0], seed=int(rid[-8:]))
        if not use_res and scen is None:         # 残差モデル・展開シナリオ予想の確率はそのまま使う（混ぜるとわずかに悪化した）
            P = sim["P"]
    except Exception as e:
        print("シミュレーション失敗（元の確率で判定）", rid, e, flush=True)
    bets = []
    for kind, rl in (("yoso", rules), ("invest", inv_rules)):
        if rl is None:
            continue
        for n, st, p, o, ev in pick_bets(P, X["O"][0], bank, rl):
            i, j, k = COMBOS[n]
            bets.append(dict(date=date, race_id=rid, venue=venue, race_no=int(rid[-4:]),
                             deadline=dl.strftime("%H:%M") if dl is not None else "",
                             combo=f"{cars[i]}-{cars[j]}-{cars[k]}", stake=st, odds=o,
                             p=round(p, 5), ev=round(ev, 3), taken_at=t.strftime("%H:%M:%S"),
                             kind=kind))
    rec, why, cand, pats = None, "", {}, {}
    mi, mi_why = None, ""
    skip_reason = ""
    try:                                         # 券種の使い分け（2車単・3連複・3連単）
        import importlib
        from . import bettypes as _bt
        importlib.reload(_bt)               # 実行中に push された選び方の変更もすぐ効かせる（git pull は push_now で毎回している）
        plans, recommend, confident_tri, load_rules = _bt.plans, _bt.recommend, _bt.confident_tri, _bt.load_rules
        tr = load_rules()
        sp = kdreams.soup(html)
        _ox, _ot, _oq = kdreams.parse_odds_exacta(sp, rid), kdreams.parse_odds_trio(sp, rid), kdreams.parse_odds_quinella(sp, rid)
        for nm, df in (("exacta", _ox), ("trio", _ot), ("quinella", _oq)):   # 券種どうしのズレを判定時点のオッズで検証するため保存
            if len(df):
                fo = day / f"odds_live_{nm}.csv.gz"
                df.assign(taken_at=t.strftime("%H:%M:%S")).to_csv(fo, mode="a", index=False, compression="gzip", header=not fo.exists())
        exo = dict(_ox[["combo", "odds"]].values)
        tro = dict(_ot[["combo", "odds"]].values)
        qno = dict(_oq[["combo", "odds"]].values)
        tri_o = {c: float(o) for c, o in od[["combo", "odds"]].values} if not od.empty else {}
        pl, win, pats = plans(P, cars, COMBOS, exo, tro, tr, tri_odds=tri_o, qn_odds=qno)
        tconf = confident_tri(pl, tr)
        base = dict(date=date, race_id=rid, venue=venue, race_no=int(rid[-4:]),
                    deadline=dl.strftime("%H:%M") if dl is not None else "", taken_at=t.strftime("%H:%M:%S"))
        for kind, plan in pl.items():
            for b in plan["bets"]:
                bets.append(dict(base, **b, kind=kind))
        if tconf is not None:                    # 自信ありの3連単：フォーメーションを確率上位の買い目に差し替え
            bets = [b for b in bets if b["kind"] != "yoso"] + [dict(base, **b, kind="yoso") for b in tconf["bets"]]
        rec, why, cand = recommend(pl, [b for b in bets if b["kind"] == "yoso"], win, tr, tri_conf=tconf)
        if rec is None:                          # どの券種も釣り合わない → 見送り（買い目を出さない）
            skip_reason = why
            bets = []
        for b in bets:
            b["rec"] = b["kind"] == rec
    except Exception as e:
        print("券種の判定失敗", rid, e, flush=True)
    tk_note = None
    try:                                         # 展開予想：主導権を取りそうなラインの「先頭→番手」2車単を紙上で記録（見送りとは別枠）
        from . import tenkai
        sp2 = kdreams.soup(html)
        exo2 = dict(kdreams.parse_odds_exacta(sp2, rid)[["combo", "odds"]].values)
        tk_note, tkb = tenkai.race_plan(ent, exo2, names=dict(zip(ent["car"].astype(int), ent["name"])))
        base2 = dict(date=date, race_id=rid, venue=venue, race_no=int(rid[-4:]),
                     deadline=dl.strftime("%H:%M") if dl is not None else "", taken_at=t.strftime("%H:%M:%S"))
        for b in tkb:                            # p は主導権確率 → モデルの2車単確率（的中率の見込み）に置き換え、主導権確率は lead に
            a2, b2 = (int(x) for x in b["combo"].split("-"))
            pe = float(sum(P[n] for n, (i, j, k) in enumerate(COMBOS) if cars[i] == a2 and cars[j] == b2))
            b.update(lead=b["p"], p=round(pe, 5), ev=round(pe * b["odds"], 3))
        bets += [dict(base2, **b, kind="tenkai", rec=False) for b in tkb]
    except FileNotFoundError:
        pass
    except Exception as e:
        print("展開予想失敗", rid, e, flush=True)
    try:                                         # 券種どうしのズレ：3連単から見て割安な3連複を紙上で記録（見送りとは別枠）
        from . import pools
        sp3 = kdreams.soup(html)
        tri3 = {c: float(o) for c, o in od[["combo", "odds"]].values} if not od.empty else {}
        tro3 = dict(kdreams.parse_odds_trio(sp3, rid)[["combo", "odds"]].values)
        base3 = dict(date=date, race_id=rid, venue=venue, race_no=int(rid[-4:]),
                     deadline=dl.strftime("%H:%M") if dl is not None else "", taken_at=t.strftime("%H:%M:%S"))
        bets += [dict(base3, **b, kind="pool", rec=False) for b in pools.trio_plan(tri3, tro3)]
        hb, hinfo = pools.hit_plan(P, cars, COMBOS, tro3)       # 当てる予想（3連複・荒れそうなら多点）
        bets += [dict(base3, **b, kind="hit4", rec=False) for b in hb]
    except Exception as e:
        print("券種ズレの判定失敗", rid, e, flush=True)
    if P_res is not None:                        # 残差モデルの期待値買い（3連複・2車複）：紙上の記録だけ（見送りとは別枠）
        try:
            from . import evbuy
            sp4 = kdreams.soup(html)
            tro4 = dict(kdreams.parse_odds_trio(sp4, rid)[["combo", "odds"]].values)
            qno4 = dict(kdreams.parse_odds_quinella(sp4, rid)[["combo", "odds"]].values)
            b3, b2 = evbuy.plan(P_res, cars, COMBOS, tro4, qno4)
            base4 = dict(date=date, race_id=rid, venue=venue, race_no=int(rid[-4:]),
                         deadline=dl.strftime("%H:%M") if dl is not None else "", taken_at=t.strftime("%H:%M:%S"))
            bets += [dict(base4, **b, kind="evr3", rec=False) for b in b3]
            bets += [dict(base4, **b, kind="evr2", rec=False) for b in b2]
            # オッズ上限なしの追加分（evr3x・evr2x）は 2026-11-24 に停止：夢のような高配当（数百〜数千倍）ばかりで実用にならないため
        except Exception as e:
            print("残差モデルの期待値買い失敗", rid, e, flush=True)
    try:                                         # 独自指標の圧縮モデル（2026/11 追加）：補正した確率で期待値1以上の2車複を紙上で記録（見送りとは別枠）
        cp = _compress_predictor()
        if cp is not None and not od.empty:
            from .compress import quinella_plan
            pred, tri_m = cp.predict(dict(race_id=rid, date=date, venue=venue, race_no=int(rid[-4:]), grade=None,
                                          bank_len=kdreams.BANK.get(venue, 400), wind=info.get("wind"),
                                          weather=info.get("weather")), ent, od)
            qno5 = dict(kdreams.parse_odds_quinella(kdreams.soup(html), rid)[["combo", "odds"]].values)
            base5 = dict(date=date, race_id=rid, venue=venue, race_no=int(rid[-4:]),
                         deadline=dl.strftime("%H:%M") if dl is not None else "", taken_at=t.strftime("%H:%M:%S"))
            bets += [dict(base5, **b, kind="cmp2", rec=False) for b in quinella_plan(pred, tri_m, qno5)]
    except Exception as e:
        print("圧縮モデルの判定失敗", rid, e, flush=True)
    try:                                         # 予想の軸をそろえる：当てる予想の上位車どうし以外の目は「_z」（紙上の記録だけ）に回して表示しない
        import re
        from . import pools
        core = pools.axis_cars(P, cars, COMBOS)
        for b in bets:
            if b["kind"] in AXIS_KINDS and not {int(x) for x in re.split("[-=]", str(b["combo"]))} <= core:
                b["kind"], b["rec"] = b["kind"] + "_z", False
        if rec and not any(b["kind"] == rec for b in bets):
            rec, why = None, ""
            skip_reason = "予想の軸（" + "・".join(map(str, sorted(core))) + "）から外れた買い目しかないため見送り"
    except Exception as e:
        print("軸そろえ失敗", rid, e, flush=True)
    scen_notes = []
    try:                                         # おすすめ＝券種をまたいで1つ（1点勝負なし。1000回のシミュレーションで一番多く当たる買い方）
        import importlib
        from . import bettypes as _bt
        from .scenario import guard_stakes
        importlib.reload(_bt)
        spm = kdreams.soup(html)
        try:
            _ow = kdreams.parse_odds_wide(spm, rid)
        except Exception as e:                   # ワイドが読めなくても他の券種で判定する
            print("ワイドのオッズを読めず", rid, e, flush=True)
            _ow = pd.DataFrame(columns=["race_id", "combo", "odds"])
        if len(_ow):                             # ワイドの判定時オッズも保存（あとで検証するため）
            fo = day / "odds_live_wide.csv.gz"
            _ow.assign(taken_at=t.strftime("%H:%M:%S")).to_csv(fo, mode="a", index=False, compression="gzip", header=not fo.exists())
        mo = dict(qn=dict(kdreams.parse_odds_quinella(spm, rid)[["combo", "odds"]].values),
                  ex=dict(kdreams.parse_odds_exacta(spm, rid)[["combo", "odds"]].values),
                  trio=dict(kdreams.parse_odds_trio(spm, rid)[["combo", "odds"]].values),
                  wide=dict(_ow[["combo", "odds"]].values) if len(_ow) else {},
                  tri={c: float(o) for c, o in od[["combo", "odds"]].values} if not od.empty else {})
        drift = load_rules_drift()
        fin = _bt.sim_finishes(sim, cars, COMBOS) if sim is not None else None
        mp = _bt.multi_plans(P, cars, COMBOS, mo, drift=drift, finishes=fin)
        pick, mwhy, msumm = _bt.multi_pick(mp)
        mi, mi_why = None, ""
        KIND_OF = dict(qn="qn", wide="wide", ex="ex", trio="trio", tri="yoso")
        shown = ("yoso", "ex", "trio", "qn", "wide")
        bets = [b for b in bets if b["kind"] not in shown + tuple(k + "_z" for k in shown)]
        for b in bets:
            b["rec"] = False
        cand = {KIND_OF[k]: v for k, v in msumm.items()}
        base_m = dict(date=date, race_id=rid, venue=venue, race_no=int(rid[-4:]),
                      deadline=dl.strftime("%H:%M") if dl is not None else "", taken_at=t.strftime("%H:%M:%S"))
        try:
            gmin = float(_bt.load_rules().synth_floor)
        except Exception:
            gmin = 1.7
        tg = guard_stakes([dict(b, stake=0) for b in pick["bets"]], budget=max(3000, 200 * pick["k"]),
                          min_synth=gmin) if pick else None   # どの目が当たっても 投資額×gmin 以上が戻る配分に
        if pick is None:
            rec, why, skip_reason = None, "", mwhy
        elif tg is None:                         # どう配分しても、当たって損する目が残る
            rec, why, skip_reason = None, "", f"{_bt.MULTI_LABEL[pick['t']]}{pick['k']}点はどの目が当たっても{gmin:g}倍以上戻る配分にできないため見送り"
        else:
            kd = KIND_OF[pick["t"]]
            bets += [dict(base_m, **b, kind=kd, rec=True) for b in tg]
            rec, why, skip_reason = kd, mwhy, ""
            scen_notes.append(f"券種をまたいで選択：{mwhy.split('（')[0]}")
        if rec is None:                          # 見送りでも予想は必ず出す（「見送り推奨」として記録。収支には入れない）
            fp = pick or (_bt.multi_pick(mp, ev_min=0, synth_min=0)[0] if len(mp) else None)
            if fp is not None:
                fb = [dict(b, stake=0) for b in fp["bets"]]
                mg = None
                for ms in (gmin, 1.0):
                    mg = guard_stakes(fb, budget=max(3000, 200 * fp["k"]), min_synth=ms)
                    if mg:
                        break
                mg = mg or [dict(b, stake=100) for b in fb]
                mi = KIND_OF[fp["t"]]
                bets += [dict(base_m, **b, kind=mi + "_m", rec=False) for b in mg]
                mi_why = (f"{_bt.MULTI_LABEL[fp['t']]}{fp['k']}点：1000回のシミュレーションで{fp['sim_hit']:.0%}的中"
                          f"・合成{fp['synth']:.1f}倍・見込み回収{fp['ev']:.0%}")
        if scen is not None:
            ll = scen.lead_lines(X["T"], 0)
            nm = dict(zip(ent["car"].astype(int), ent["name"].astype(str)))
            scen_notes.append("主導権を取るライン：" + "、".join(
                f"{'-'.join(map(str, c))}（{nm.get(c[0], '')}）{p:.0%}" for c, p in ll[:3]))
    except Exception as e:
        print("券種をまたいだ買い目作成失敗", rid, e, flush=True)
    try:                                         # 一言解説（買い目の有無にかかわらず記録）
        from .insights import race_notes
        notes, line_probs = race_notes(P, X["T"], ent, his, od, date=date, venue=venue)
    except Exception as e:
        print("解説作成失敗", rid, e, flush=True); notes, line_probs = [], []
    if tk_note:
        notes = [tk_note] + notes
    notes = scen_notes + notes
    sim_ev = None
    if sim is not None:                          # 出す買い目を1000回のレースで採点
        try:
            from .simulate import evaluate, lead_lines, summary_text
            shown = ([b for b in bets if b.get("rec")] or [b for b in bets if b["kind"].endswith("_m")]
                     or [b for b in bets if b["kind"] == "yoso"])
            sim_ev = evaluate(sim, cars, shown)
            notes = summary_text(sim_ev, lead_lines(sim, X["T"], 0, cars)) + notes
        except Exception as e:
            print("シミュレーション採点失敗", rid, e, flush=True)
    with open(day / "notes.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(race_id=rid, venue=venue, race_no=int(rid[-4:]),
                                deadline=dl.strftime("%H:%M") if dl is not None else "",
                                notes=notes, lines=line_probs,
                                names={int(c): str(nm) for c, nm in zip(ent["car"], ent["name"])},
                                n_bets=sum(b["kind"] in ("yoso", "ex", "trio", "qn", "wide") for b in bets),
                                n_invest=sum(b["kind"] == "invest" for b in bets),
                                rec=rec, rec_reason=why, sim=sim_ev,
                                mi=mi, mi_reason=mi_why,
                                cand={k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in cand.items()},
                                reason=skip_reason or ("" if rec or any(b["kind"] == "yoso" for b in bets) else "当たる確率の高い目で組んでも合成オッズが低すぎるため")),
                           ensure_ascii=False) + "\n")
    bets = _cap_odds(bets)
    if bets:
        f = day / "bets.csv"
        new = pd.DataFrame(bets)
        if f.exists():                   # 列構成が変わっても壊れないよう、全体を読み直して書く
            old = pd.read_csv(f, dtype={"race_id": str}, on_bad_lines="skip")
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(f, index=False)
    return bets, ent, notes


def load_rules_drift():
    """判定時オッズ→確定時の見込み（config.json の type_rules.odds_drift。無ければ 0.93）"""
    try:
        from .bettypes import load_rules
        return float(load_rules().odds_drift)
    except Exception:
        return 0.93


def _fmt_block(bets, names):
    from .strategy import summary, formations
    sm = summary([(0, b["stake"], b["p"], b["odds"], b["ev"]) for b in bets])
    L = [f"**{len(bets)}点 {sm['total']:,}円**／合成オッズ {sm['synth']:.1f}倍／的中率 {sm['hit_p']:.0%}／期待値 {sm['race_ev']:.2f}"
         + ("　⚠妙味薄" if sm["synth"] < 2.0 else ""), ""]
    order = {}
    for b in bets:
        f = int(b["combo"].split("-")[0]); order[f] = order.get(f, 0) + b["p"]
    rows = formations([b["combo"] for b in bets])
    for f in sorted({r[0] for r in rows}, key=lambda f: -order.get(f, 0)):
        L.append(f"**軸 {f} {names.get(f, '')}**")
        for a, bs, cs in [r for r in rows if r[0] == f]:
            n = len(bs) * len(cs) - sum(1 for x in bs for y in cs if x == y)
            L.append(f"- {a} → {'・'.join(map(str, bs))} → {'・'.join(map(str, cs))}（{n}点）")
        L.append("")
    L.append("<details><summary>金額の内訳</summary>\n")
    L += ["| 買い目 | 金額 | オッズ |", "|---|---:|---:|"]
    L += [f"| {b['combo']} | {b['stake']:,}円 | {b['odds']:.1f} |" for b in sorted(bets, key=lambda b: b["odds"])]
    L += ["", "</details>", ""]
    return L


def fmt(venue, rid, dl, bets, ent, mode, notes=None):
    """通知はシステムが選んだ1つの買い方だけ（券種・点数の選択肢は出さない）"""
    names = dict(zip(ent["car"], ent["name"]))
    head = f"### {kdreams.VENUE_JA.get(venue, venue)} {int(rid[-4:])}R（締切 {dl:%H:%M}）"
    L = [head + ("【紙上運用】" if mode == "paper" else ""), ""]
    h4 = [b for b in bets if b.get("kind") == "hit4"]
    if h4:
        hp = sum(b["p"] for b in h4); st = sum(b["stake"] for b in h4)
        L += [f"#### ◎当てる予想：3連複 {len(h4)}点 {st:,}円（的中率 {hp:.0%}・合成 {h4[0].get('synth', 0):.1f}倍）", ""]
        L += [f"- {b['combo']}　{b['stake']:,}円（{b['odds']:.1f}倍）" for b in sorted(h4, key=lambda b: -b["p"])] + [""]
    rec = next((b["kind"] for b in bets if b.get("rec")), None)
    if rec in ("ex", "trio", "qn", "wide"):
        lab = {"ex": "2車単", "trio": "3連複", "qn": "2車複", "wide": "ワイド"}[rec]
        rb = [b for b in bets if b.get("kind") == rec]
        hp = sum(b["p"] for b in rb); st = sum(b["stake"] for b in rb)
        L += [f"#### ◎予想：{lab} {len(rb)}点 {st:,}円（的中率 {hp:.0%}）", ""]
        L += [f"- {b['combo'].replace('=', '-')}　{b['stake']:,}円（{b['odds']:.1f}倍）" for b in sorted(rb, key=lambda b: -b["p"])] + [""]
    else:
        yoso = [b for b in bets if b.get("kind", "yoso") == "yoso"]
        if yoso:
            L += ["#### ◎予想：3連単"] + _fmt_block(yoso, names)
        elif not any(b.get("kind") in ("tenkai", "pool", "hit4") for b in bets):
            return None
        else:
            L += ["#### ◎予想：見送り（条件を満たす買い方がないため）", ""]
    pb = [b for b in bets if b.get("kind") == "pool"]
    if pb:
        L += ["#### ズレ予想 3連複（紙上・検証中）"] + [f"- 3連複 {b['combo']}　{b['stake']:,}円（{b['odds']:.1f}倍）割安度 {b['ratio']:.2f}" for b in pb] + [""]
    tk = [b for b in bets if b.get("kind") == "tenkai"]
    if tk:
        L += ["#### 展開予想（紙上・検証中）"] + [f"- 2車単 {b['combo']}　{b['stake']:,}円（{b['odds']:.1f}倍）主導権確率 {b['p']:.0%}" for b in tk] + [""]
    if notes:
        L += ["**ひとこと**"] + [f"- {n}" for n in notes] + [""]
    L.append(f"判定時刻 {bets[0]['taken_at']}")
    return "\n".join(L)


def _note_fail(date, venue, rid, dl, err):
    """判定できなかったレースを、理由付きの見送りとして記録する"""
    day = LIVE / date
    day.mkdir(parents=True, exist_ok=True)
    with open(day / "notes.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(race_id=rid, venue=venue, race_no=int(rid[-4:]), deadline=dl.strftime("%H:%M"),
                                notes=[], lines=[], n_bets=0, n_invest=0,
                                reason=f"データ取得・判定に失敗したため見送り（{err}）"), ensure_ascii=False) + "\n")


def push_now(msg):
    """買い目を出すたびにリポジトリへ保存（ページへの反映を早くするため）"""
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    import subprocess
    sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
    sh('git config user.name "keirin-bot"; git config user.email "bot@users.noreply.github.com"')
    sh("git add data/live site")
    if sh(f'git commit -qm "{msg}"').returncode != 0:
        return
    for _ in range(3):
        if sh("git pull -q --rebase -X theirs && git push -q").returncode == 0:
            return
        time.sleep(5)
    print("保存失敗（実行の最後にまとめて保存します）", flush=True)


def _code_sig():
    """読み込んでいるコード（keirin_ev/*.py・main.py・config.json）の指紋。途中で変わったら再起動して新しいコードで続ける"""
    import hashlib
    h = hashlib.sha1()
    for f in sorted(Path("keirin_ev").glob("*.py")) + [Path("main.py"), Path("config.json")]:
        try:
            h.update(f.name.encode()); h.update(f.read_bytes())
        except Exception:
            pass
    return h.hexdigest()


def _reload_if_changed(sig, t0, max_minutes, last_pull):
    """数分おきにmainの最新を取り込み、コードが変わっていたら残り時間で自分を起動し直す（判定済みのレースは notes.jsonl から引き継ぐ）"""
    if os.environ.get("GITHUB_ACTIONS") and time.time() - last_pull[0] > 180:
        import subprocess
        subprocess.run("git -c rebase.autoStash=true pull -q --rebase -X theirs", shell=True, capture_output=True)
        last_pull[0] = time.time()
    if _code_sig() == sig:
        return
    left = int(max_minutes - (time.time() - t0) / 60)
    if left < 2:
        return
    print(f"{now():%H:%M} コードが更新されたので新しいコードで再起動（残り{left}分）", flush=True)
    import sys
    os.execv(sys.executable, [sys.executable, "main.py", "live", str(left)])


def run(max_minutes=340):
    t0 = time.time()
    sig, last_pull = _code_sig(), [time.time()]
    date = now().strftime("%Y-%m-%d")
    if now().hour < 4:                         # 日付をまたいだミッドナイト
        date = (now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cfg, rules = load_config()
    inv_rules = load_invest_rules(cfg)
    mode, bank = cfg.get("mode", "paper"), cfg.get("bankroll", 100_000)
    try:
        model = TrifectaModel.from_json("models/latest.json")
    except Exception as e:
        print("モデルなし（オッズの記録のみ行います）:", e); model = None
    if strategy() != "hit":
        _compress_predictor()                  # 圧縮モデルの過去データを先に読んでおく（最初の判定が締切に遅れないように）
    kdreams.SLEEP_SEC = 1.0
    full = schedule(date)
    sched = [x for x in full if x[0] > now() + pd.Timedelta(minutes=LEAD_MIN)]
    print(f"{date}: 判定対象 {len(sched)} レース", flush=True)
    try:                                   # 今日の締切一覧をページへ
        day = LIVE / date; day.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([dict(venue=v, race_no=int(r[-4:]), race_id=r, deadline=d.strftime("%H:%M"))
                      for d, v, r in full]).to_csv(day / "schedule.csv", index=False)
        from .site import export_live
        export_live(date)
        push_now("live 締切一覧")
    except Exception as e:
        print("締切一覧の保存失敗:", e, flush=True)
    note = Notifier(date)
    done, fails = set(), {}
    nf = LIVE / date / "notes.jsonl"          # 前の実行で判定済みのレースは判定し直さない
    if nf.exists():
        for line in nf.read_text(encoding="utf-8").splitlines():
            try:
                done.add(str(json.loads(line)["race_id"]))
            except Exception:
                pass
    sched = [x for x in sched if x[2] not in done]
    have = set()
    rf = LIVE / date / "results.csv"
    if rf.exists():
        have = set(pd.read_csv(rf, dtype={"race_id": str})["race_id"])
    # 結果待ち：締切を過ぎた（これから過ぎる）全レース。今日のバンク傾向にも使う
    pending = {rid: [dl, venue, dl + pd.Timedelta(minutes=RES_AFTER)]
               for dl, venue, rid in full if rid not in have}
    from . import late
    late_done = late.done_ids(date)            # 締切直前のオッズ：締切がまだ先の全レース（前の実行で記録済みは除く）
    late_q = [x for x in full if x[0] > now() + pd.Timedelta(minutes=late.LATE_MIN) and x[2] not in late_done]
    if not sched:
        print("判定するレースが残っていません", flush=True)
    while (sched or pending or late_q) and time.time() - t0 < max_minutes * 60:
        _reload_if_changed(sig, t0, max_minutes, last_pull)
        got = check_results(date, pending, lambda m: print(m, flush=True))
        if got:
            print(f"{now():%H:%M} 結果取り込み {len(got)}レース", flush=True)
            try:
                from .site import export_live
                export_live(date)
            except Exception as e:
                print("サイト用データ更新失敗:", e, flush=True)
            push_now(f"live 結果 {len(got)}R")
        t = now()
        for dl, venue, rid in late_q:              # 締切直前のオッズを記録（判定後の下落を実測するため。買い目には使わない）
            left = (dl - t).total_seconds() / 60
            if rid in late_done or left > late.LATE_MAX:
                continue
            late_done.add(rid)                     # 失敗しても取り直さない（判定の時間を削らないため）
            if left < late.LATE_MIN:
                continue
            try:
                n = late.snapshot(date, venue, rid, dl)
                print(f"{now():%H:%M:%S} {venue} {int(rid[-4:])}R 直前オッズ {n}券種（締切{left:.1f}分前）", flush=True)
            except Exception as e:
                print("直前オッズの取得失敗", rid, e, flush=True)
        late_q = [x for x in late_q if x[2] not in late_done]
        t = now()
        for dl, venue, rid in sched:
            left = (dl - t).total_seconds() / 60
            if rid in done or left > LEAD_MAX:
                continue
            if left < LEAD_MIN:
                done.add(rid)
                if fails.get(rid):                 # 何度やっても判定できなかった → 理由付きで見送りにしてページへ
                    _note_fail(date, venue, rid, dl, fails[rid])
                    try:
                        from .site import export_live
                        export_live(date)
                    except Exception as e:
                        print("サイト用データ更新失敗:", e, flush=True)
                    push_now(f"live {venue} {int(rid[-4:])}R 判定失敗")
                continue
            try:
                bets, ent, notes = judge(date, venue, rid, model, rules, bank, dl, inv_rules)
            except Exception as e:                 # 失敗しても締切の LEAD_MIN 分前までは30秒ごとにやり直す
                import traceback
                fails[rid] = f"{type(e).__name__}: {e}"[:200]
                print("判定失敗（やり直します）", rid, fails[rid], flush=True)
                traceback.print_exc()
                continue
            done.add(rid)
            print(f"{t:%H:%M} {venue} {int(rid[-4:])}R 締切{dl:%H:%M} 買い目{len(bets)}点", flush=True)
            msg = fmt(venue, rid, dl, bets, ent, mode, notes) if bets else None
            if msg:
                note.post(msg)
            try:                           # 買い・見送りにかかわらずページを更新
                from .site import export_live
                export_live(date)
            except Exception as e:
                print("サイト用データ更新失敗:", e, flush=True)
            push_now(f"live {venue} {int(rid[-4:])}R {'買い' if bets else '見送り'}")
        sched = [x for x in sched if x[2] not in done]
        time.sleep(30)
    print("終了", flush=True)


# ---------- 当日の結果（レース後すぐ取り込む） ----------
def fetch_result(date, venue, rid):
    """締切後のレースページから結果を読み、data/live/日付/ に追記する。取れたら True"""
    html = kdreams.fetch(f"/{venue}/racedetail/{rid}/", refresh=True)
    ent, _, res, _, fin, _ = kdreams.parse_race(html, rid, extra=True)
    if not res:
        return False
    day = LIVE / date
    day.mkdir(parents=True, exist_ok=True)
    _append(day / "results.csv", pd.DataFrame([res]))
    # 今日のバンク傾向（perspectives.today_bias）用：勝者のライン位置
    w = ent[ent["car"] == res["first"]]
    if len(w):
        w = w.iloc[0]
        kim = fin.loc[fin["car"] == res["first"], "kimarite"] if len(fin) and "kimarite" in fin else []
        _append(day / "today.csv", pd.DataFrame([dict(
            race_id=rid, venue=venue, race_no=int(rid[-4:]), first=res["first"],
            w_head=float(w["line_pos"] == 1 and (w["line_size"] or 1) > 1),
            w_2nd=float(w["line_pos"] == 2),
            kimarite=kim.iloc[0] if len(kim) else None)]))
    return True


def _append(f, df):
    if f.exists():
        old = pd.read_csv(f, dtype={"race_id": str}, on_bad_lines="skip")
        df = pd.concat([old[~old["race_id"].isin(df["race_id"].astype(str))], df], ignore_index=True)
    df.to_csv(f, index=False)


def check_results(date, pending, log=print):
    """pending: {rid: [締切, 場, 次に見る時刻]} 結果の出たレースを取り込み、取れたレースIDを返す"""
    got = []
    t = now()
    for rid, (dl, venue, nxt) in list(pending.items()):
        if t < nxt:
            continue
        if t > dl + pd.Timedelta(minutes=RES_GIVEUP):
            pending.pop(rid); continue
        try:
            ok = fetch_result(date, venue, rid)
        except Exception as e:
            log(f"結果取得失敗 {rid} {e}"); ok = False
        if ok:
            pending.pop(rid); got.append(rid)
        else:
            pending[rid][2] = t + pd.Timedelta(minutes=RES_EVERY)
    return got


# ---------- 紙上運用の成績 ----------
def paper_results():
    from . import store as st
    bets = []
    for f in sorted(LIVE.glob("*/bets.csv")):
        bets.append(pd.read_csv(f, dtype={"race_id": str}))
    if not bets:
        return None
    b = pd.concat(bets, ignore_index=True)
    if "kind" not in b:
        b["kind"] = np.nan
    b["kind"] = b["kind"].fillna("invest")       # 区分導入前の買い目は期待値型
    res = []
    for d in b["date"].unique():
        f = st.day_dir(d) / "results.csv.gz"
        lf = LIVE / str(d) / "results.csv"          # 確定データが来るまでは当日取り込んだ結果を使う
        got = st.read(str(d), "results") if f.exists() else pd.DataFrame()   # 空のファイルでも止まらない
        if got.empty and lf.exists():
            try:
                got = pd.read_csv(lf, dtype={"race_id": str})
            except pd.errors.EmptyDataError:
                got = pd.DataFrame()
        if not got.empty:
            res.append(got)
    if not res:
        return b.assign(hit=np.nan, payout=np.nan)
    r = pd.concat(res)
    r["win"] = r["first"].astype(str) + "-" + r["second"].astype(str) + "-" + r["third"].astype(str)
    for c in ("ex_combo", "ex_payout", "trio_combo", "trio_payout", "qn_combo", "qn_payout", "wide_combos", "wide_payouts"):
        if c not in r:
            r[c] = np.nan
    b = b.merge(r[["race_id", "win", "tri_payout", "ex_combo", "ex_payout", "trio_combo", "trio_payout",
                   "qn_combo", "qn_payout", "wide_combos", "wide_payouts"]], on="race_id", how="left")
    is_ex, is_trio, is_qn = b["kind"].isin(["ex", "tenkai", "ex_z", "tenkai_z", "ex_m"]), b["kind"].isin(["trio", "pool", "hit4", "trio_z", "evr3", "evr3x", "trio_m"]), b["kind"].isin(["qn", "qn_z", "evr2", "evr2x", "cmp2", "qn_m"])   # tenkai は2車単、pool・evr3 は3連複、evr2・cmp2 は2車複
    target = np.where(is_ex, b["ex_combo"], np.where(is_trio, b["trio_combo"],
                      np.where(is_qn, b["qn_combo"], b["win"])))
    pay = np.where(is_ex, b["ex_payout"], np.where(is_trio, b["trio_payout"],
                   np.where(is_qn, b["qn_payout"], b["tri_payout"])))
    b["hit"] = (b["combo"] == target).astype(object)
    b["payout"] = np.where(b["hit"], b["stake"] / 100 * pd.to_numeric(pay, errors="coerce"), 0)
    b.loc[pd.isna(target), ["hit", "payout"]] = np.nan
    is_w = b["kind"].isin(["wide", "wide_z", "wide_m"])  # ワイドは当たりが3組（同着ならそれ以上）
    if is_w.any():
        def _wide(row):
            cs, ps = row["wide_combos"], row["wide_payouts"]
            if not isinstance(cs, str) or not cs:
                return np.nan, np.nan
            pay = dict(zip(cs.split("|"), str(ps).split("|")))
            if row["combo"] in pay:
                return True, row["stake"] / 100 * float(pay[row["combo"]])
            return False, 0.0
        hp = b.loc[is_w].apply(_wide, axis=1, result_type="expand")
        b.loc[is_w, "hit"] = hp[0].values
        b.loc[is_w, "payout"] = hp[1].values
    if "rec" not in b:
        b["rec"] = False
    return b
