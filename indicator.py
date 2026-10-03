# =====================================================================
# TREND INDICATOR — BTC + ETH  (version finale)
#   Signal   : prix > MA200 -> investi ; sinon cash
#   Risque   : vol-target 40 % par actif, plafond 100 %, cash rémunéré (FRED DTB3)
#   Règles   : ajouts le lundi (bande 10 %), sorties possibles tous les jours
#   Sortie   : docs/data.js  (lu par docs/index.html)
#   Alerte   : message Telegram si un ordre est à passer (optionnel)
# Variables d'environnement : FRED_API_KEY (+ TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
# Lancer : python indicator.py   (bougie du jour UTC ignorée : elle est incomplète)
# =====================================================================
import os
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------ CONFIG -------------------------------
START = "2014-10-01"
SPLIT = "2025-12-31"              # train <= 2025 ; test = 2026
WARM_DAYS = 420
ANN = 365
ASSETS = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum"}

CFG = dict(
    ma=200,
    target_vol=0.40, max_lev=1.0,
    rebal_band=0.10, daily_exit=True,
    cash=True,
    fee_bps=10, slip_bps=5,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(HERE, "docs", "data.js")


# ------------------------------ DONNÉES ------------------------------
def load_yf(sym, tries=3):
    import yfinance as yf
    d = None
    for k in range(tries):
        try:
            d = yf.download(sym, start=START, auto_adjust=True, progress=False)
            if d is not None and len(d) > 300:
                break
        except Exception as e:
            print(f"yfinance {sym} essai {k + 1} : {e}")
        time.sleep(5 * (k + 1))
    else:
        raise RuntimeError(f"yfinance : pas de données pour {sym}")
    d.columns = [(c[0] if isinstance(c, tuple) else c).lower() for c in d.columns]
    d = d[["high", "low", "close"]].dropna()
    idx = pd.DatetimeIndex(d.index)
    d.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    return d[~d.index.duplicated()].sort_index()


def fred(series_id, key):
    import requests
    r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                     params=dict(series_id=series_id, api_key=key, file_type="json",
                                 observation_start="2013-01-01"), timeout=60)
    j = r.json()
    if "observations" not in j:
        raise RuntimeError(f"FRED {series_id}: {str(j)[:200]}")
    s = pd.Series({pd.Timestamp(o["date"]): pd.to_numeric(o["value"], errors="coerce")
                   for o in j["observations"]})
    return s.dropna().sort_index()


def daily_cash_rate(idx):
    """Taux 3 mois (%/an) -> rendement quotidien, décalé de 2 jours (info déjà publiée)."""
    try:
        key = os.environ.get("FRED_API_KEY", "")
        if not key:
            raise RuntimeError("clé FRED absente")
        r = fred("DTB3", key).reindex(idx, method="ffill").shift(2)
        print(f"Cash : taux 3 mois FRED chargé (moyenne {r.mean():.2f} %/an).")
        return (r / 100 / ANN).fillna(0.0)
    except Exception as e:
        print("Taux cash indisponible -> 0 % :", e)
        return pd.Series(0.0, index=idx)


# ------------------------------ SIGNAL + BACKTEST ----------------------
def exposure(df, cfg):
    """1 si clôture > MA200, sinon 0. NaN pendant la chauffe."""
    c = df["close"]
    ma = c.rolling(cfg["ma"]).mean()
    s = (c > ma).astype(float)
    s[ma.isna()] = np.nan
    return s


def run(df, expo, cash_d, cfg):
    """Décision à la clôture t, appliquée au rendement t+1 : pas de look-ahead."""
    close = df["close"]
    ret = close.pct_change().fillna(0).values
    valid = expo.notna().values
    rv = pd.Series(ret).ewm(span=30).std().values * np.sqrt(ANN)
    size = np.nan_to_num(np.clip(cfg["target_vol"] / np.where(rv > 0, rv, np.nan), 0, cfg["max_lev"]))
    tgt = np.nan_to_num(expo.values) * size
    reb = close.index.dayofweek == 0                       # entrées / ajouts : lundi
    w, cur = np.zeros(len(close)), 0.0
    for t in range(len(close)):
        if valid[t]:
            zero_out = tgt[t] == 0 and cur > 0
            if reb[t]:
                if abs(tgt[t] - cur) > cfg["rebal_band"] or zero_out:
                    cur = tgt[t]
            elif cfg["daily_exit"] and (tgt[t] < cur - cfg["rebal_band"] or zero_out):
                cur = tgt[t]                                # sorties rapides, tous les jours
        w[t] = cur
    w_lag = np.r_[0.0, w[:-1]]
    turn = np.abs(np.diff(np.r_[0.0, w_lag]))
    cash = (1 - w_lag) * cash_d.reindex(close.index).fillna(0).values if cfg["cash"] else 0.0
    net = w_lag * ret + cash - turn * (cfg["fee_bps"] + cfg["slip_bps"]) / 1e4
    return (pd.Series(net, index=close.index), pd.Series(w_lag, index=close.index), tgt,
            pd.Series(w, index=close.index))


def metrics(r):
    r = pd.Series(r).dropna()
    if len(r) < 30 or r.std() == 0:
        return dict(sharpe=0.0, cagr=0.0, mdd=0.0, calmar=0.0)
    eq = (1 + r).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (ANN / len(r)) - 1
    return dict(sharpe=float(r.mean() / r.std() * np.sqrt(ANN)), cagr=float(cagr), mdd=float(mdd),
                calmar=float(cagr / abs(mdd)) if mdd < 0 else 0.0)


def run_all(data, cash_d):
    out = {}
    for s, d in data.items():
        out[s] = run(d, exposure(d, CFG), cash_d.reindex(d.index).fillna(0.0), CFG)
    return out


def port_returns(out):
    return pd.DataFrame({s: v[0].iloc[WARM_DAYS:] for s, v in out.items()}).mean(axis=1, skipna=True)


def weights_table(out, n):
    W = pd.DataFrame({s: v[3] for s, v in out.items()}) / n         # poids décidés, part du capital
    W["cash"] = 1 - W.sum(axis=1)
    return W


# ------------------------------ EXPORT ---------------------------------
def clean(o):
    """NaN / inf -> None, types numpy -> types Python (JSON valide)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def rl(x, nd=4):
    return [None if not np.isfinite(v) else round(float(v), nd) for v in x]


def build_payload(data, cash_d):
    n = len(data)
    out = run_all(data, cash_d)
    W = weights_table(out, n)
    r = port_returns(out)
    bh = pd.DataFrame({s: d["close"].pct_change().iloc[WARM_DAYS:] for s, d in data.items()}).mean(axis=1, skipna=True)
    bh = bh.reindex(r.index).fillna(0.0)

    eq, bq = (1 + r).cumprod() * 100, (1 + bh).cumprod() * 100
    last = r.index[-1]

    def tot(x):
        return float((1 + x).prod() - 1)

    def per(x):
        return [float(x.iloc[-1]), tot(x.iloc[-7:]), tot(x.iloc[-30:]),
                tot(x[x.index.year == last.year]), tot(x.iloc[-365:]), tot(x)]

    labels = ["Dernier jour", "7 jours", "30 jours", "Depuis le 1er janvier", "12 mois", "Depuis le début"]
    periods = [[l, a, b] for l, a, b in zip(labels, per(r), per(bh))]

    def block(x):
        return {"full": metrics(x), "train": metrics(x[x.index <= SPLIT]), "test": metrics(x[x.index > SPLIT])}

    mon = r.groupby([r.index.year, r.index.month]).apply(lambda x: float((1 + x).prod() - 1))
    monthly = {f"{y}-{m:02d}": v for (y, m), v in mon.items()}

    assets = []
    for s, d in data.items():
        c = d["close"]
        ma = c.rolling(CFG["ma"]).mean()
        state = c > ma
        grp = state.ne(state.shift()).cumsum()
        streak = int((grp == grp.iloc[-1]).sum())
        rv = c.pct_change().ewm(span=30).std().iloc[-1] * np.sqrt(ANN)
        tail = c.index[-365:]
        assets.append(dict(
            sym=s, name=ASSETS[s], price=float(c.iloc[-1]), ma=float(ma.iloc[-1]),
            gap=float(c.iloc[-1] / ma.iloc[-1] - 1), vol=float(rv), signal=bool(state.iloc[-1]), streak=streak,
            target=float(W[s].iloc[-1]), prev_w=float(W[s].iloc[-2]),
            chart=dict(dates=[str(t.date()) for t in tail], close=rl(c.loc[tail], 2), ma=rl(ma.loc[tail], 2))))

    trades = []
    for s in data:
        w = out[s][3] / n
        dw = w.diff().fillna(0.0)
        for t in dw[dw.abs() > 1e-9].index:
            trades.append(dict(date=str(t.date()), sym=s, frm=float(w.shift().fillna(0.0).loc[t]), to=float(w.loc[t])))
    trades.sort(key=lambda x: (x["date"], x["sym"]), reverse=True)

    wh = W.iloc[-180:]
    actions = []
    for a in assets:
        d = a["target"] - a["prev_w"]
        if abs(d * n) > CFG["rebal_band"] or (a["target"] == 0 and a["prev_w"] > 0):
            actions.append(dict(sym=a["sym"], act="ACHETER" if d > 0 else "VENDRE", delta=abs(d)))

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    last_bar = max(d.index[-1] for d in data.values())
    payload = dict(
        generated_utc=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        last_bar=str(last_bar.date()),
        stale=bool((today - last_bar).days > 2),
        config=dict(n=n, rebal_band=CFG["rebal_band"], target_vol=CFG["target_vol"], ma=CFG["ma"],
                    cost_bps=CFG["fee_bps"] + CFG["slip_bps"], split=SPLIT,
                    assets=[ASSETS[s] for s in data]),
        cash_w=float(W["cash"].iloc[-1]), exposure=float(1 - W["cash"].iloc[-1]),
        assets=assets,
        actions=actions,
        periods=periods,
        metrics=dict(strat=block(r), bh=block(bh)),
        equity=dict(dates=[str(t.date()) for t in r.index], strat=rl(eq, 2), bh=rl(bq, 2)),
        weights=dict(dates=[str(t.date()) for t in wh.index],
                     **{s: rl(wh[s], 4) for s in data}, cash=rl(wh["cash"], 4)),
        monthly=monthly,
        trades=trades[:40],
    )
    return clean(payload), actions


def notify(payload, actions):
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (tok and chat) or not actions:
        return
    try:
        import requests
        lines = [f"📈 Trend Indicator — bougie du {payload['last_bar']}"]
        for a in actions:
            lines.append(f"• {a['act']} {a['sym']} : {a['delta']:.1%} du capital")
        lines.append(f"Cash cible : {payload['cash_w']:.1%}")
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": "\n".join(lines)}, timeout=30)
    except Exception as e:
        print("Telegram :", e)


# -------------------------------- MAIN ---------------------------------
def main():
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    data = {s: load_yf(s) for s in ASSETS}
    data = {s: d[d.index < today] for s, d in data.items()}        # bougie du jour = incomplète
    cash_d = daily_cash_rate(data["BTC-USD"].index)
    payload, actions = build_payload(data, cash_d)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("window.DATA = " + json.dumps(payload, ensure_ascii=False, allow_nan=False) + ";\n")

    print(f"\nBougie du {payload['last_bar']} | exposition {payload['exposure']:.1%} | cash {payload['cash_w']:.1%}")
    for a in payload["assets"]:
        print(f"  {a['sym']}: {'HAUSSIER' if a['signal'] else 'BAISSIER'} | cible {a['target']:.1%} | précédent {a['prev_w']:.1%}")
    print("Ordres vs veille :", actions if actions else "aucun")
    print(f"Écrit : {OUT_FILE}")
    notify(payload, actions)


if __name__ == "__main__":
    main()
