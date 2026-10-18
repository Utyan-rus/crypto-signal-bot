"""
webapp.py
Flask app serving a Telegram Mini App (WebApp): a list of tracked
assets with Long/Short/Neutral reads, a detail view per asset, plus
tabs for a position-size calculator, multi-timeframe check, backtest
and price alerts. Runs in a background thread alongside the bot's
polling loop (started from bot.py).

Every mutating/data request must carry a valid Telegram WebApp
initData string (verified with the bot token) for a chat id that is
in the bot's authorized list — same invite-only boundary as the bot
itself, so the panel can't be used by someone who just finds the URL.

Read-only toward the market: never places trades, never touches a wallet.
"""

import hashlib
import hmac
import json
import os
import uuid
from functools import wraps
from urllib.parse import parse_qsl

from flask import Flask, jsonify, render_template_string, request

from analysis import analyze, backtest as run_backtest, STRATEGY_NAMES
from exchange import ExchangeClient
import storage

app = Flask(__name__)
client = ExchangeClient(os.getenv("EXCHANGE_ID", "binance"))

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
PANEL_TIMEFRAME = os.getenv("PANEL_TIMEFRAME", "1h")
MTF_TIMEFRAMES = ["15m", "1h", "4h"]

DEFAULT_PANEL_SYMBOLS = [s.strip().upper() for s in os.getenv(
    "PANEL_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT"
).split(",") if s.strip()]

# Populated by bot.py's periodic refresh job, read by the panel's list view
SIGNALS_CACHE: dict = {}


# ------------------------------------------------------------- auth ----

def validate_init_data(init_data: str):
    """Verifies a Telegram WebApp initData string per Telegram's documented
    algorithm and checks the user is in the bot's authorized chat list.
    Returns the user dict on success, None otherwise."""
    if not init_data or not BOT_TOKEN:
        return None
    try:
        parsed = dict(parse_qsl(init_data, strict_parsing=True))
    except ValueError:
        return None
    received_hash = parsed.pop("hash", None)
    if not received_hash:
        return None
    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(parsed.items()))
    secret_key = hmac.new(b"WebAppData", BOT_TOKEN.encode(), hashlib.sha256).digest()
    computed_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(computed_hash, received_hash):
        return None
    user_json = parsed.get("user")
    if not user_json:
        return None
    user = json.loads(user_json)
    state = storage.load()
    if user.get("id") not in state.get("authorized_chat_ids", []):
        return None
    return user


def require_webapp_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        init_data = request.headers.get("X-Telegram-Init-Data", "")
        user = validate_init_data(init_data)
        if not user:
            return jsonify({"error": "unauthorized"}), 401
        request.tg_user = user
        return f(*args, **kwargs)
    return wrapper


# ----------------------------------------------------------- helpers ----

def get_panel_symbols(state: dict) -> list:
    return state["panel_symbols"] if state.get("panel_symbols") else list(DEFAULT_PANEL_SYMBOLS)


def compute_signal_dict(symbol: str, timeframe: str = PANEL_TIMEFRAME, strategy: str = None):
    if strategy is None:
        strategy = storage.load().get("strategy", "guarded")
    df = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=260)
    if len(df) < 220:
        return None
    imbalance = client.fetch_order_book_imbalance(symbol)
    funding = client.fetch_funding_rate(symbol)
    signal = analyze(df, order_book_imbalance=imbalance, funding_rate=funding, strategy=strategy)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy,
        "direction": signal.direction,
        "confidence": signal.confidence,
        "risk_level": signal.risk_level,
        "entry": round(signal.entry, 4),
        "stop_loss": round(signal.stop_loss, 4),
        "take_profit": round(signal.take_profit, 4),
        "reasons": signal.reasons,
    }


# -------------------------------------------------------------- page ----

PANEL_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Signal Panel</title>
<script src="https://telegram.org/js/telegram-web-app.js"></script>
<style>
  :root {
    --bg: var(--tg-theme-bg-color, #ffffff);
    --text: var(--tg-theme-text-color, #111111);
    --hint: var(--tg-theme-hint-color, #999999);
    --link: var(--tg-theme-link-color, #2481cc);
    --button: var(--tg-theme-button-color, #2481cc);
    --button-text: var(--tg-theme-button-text-color, #ffffff);
    --secondary-bg: var(--tg-theme-secondary-bg-color, #f0f0f0);
    --green: #22c55e;
    --red: #ef4444;
    --gray: #9ca3af;
    --yellow: #eab308;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding-top: env(safe-area-inset-top, 0px);
    padding-bottom: calc(60px + env(safe-area-inset-bottom, 0px));
  }
  header { padding: 16px; font-size: 20px; font-weight: 700; border-bottom: 1px solid var(--secondary-bg); }
  .sub { color: var(--hint); font-size: 13px; font-weight: 400; margin-top: 2px; }
  .screen { display: none; padding: 12px; }
  .screen.active { display: block; }

  .card {
    display: flex; align-items: center; justify-content: space-between;
    background: var(--secondary-bg); border-radius: 14px; padding: 14px 16px;
    margin-bottom: 10px; cursor: pointer; transition: opacity .15s;
  }
  .card:active { opacity: 0.6; }
  .symbol { font-weight: 700; font-size: 16px; }
  .conf { color: var(--hint); font-size: 12px; margin-top: 2px; }
  .badge { padding: 6px 12px; border-radius: 999px; font-size: 13px; font-weight: 700; color: white; white-space: nowrap; }
  .LONG, .LONG_ALL { background: var(--green); }
  .SHORT, .SHORT_ALL { background: var(--red); }
  .NEUTRAL, .MIXED { background: var(--gray); }
  .LONG_MAJORITY { background: #86efac; color: #14532d; }
  .SHORT_MAJORITY { background: #fca5a5; color: #7f1d1d; }
  .risk-LOW { color: var(--green); }
  .risk-MEDIUM { color: var(--yellow); }
  .risk-HIGH { color: var(--red); }

  .remove-btn {
    width: 28px; height: 28px; border-radius: 50%; border: none;
    background: var(--bg); color: var(--hint); font-size: 16px; margin-left: 8px;
  }
  .add-row { display: flex; gap: 8px; margin-bottom: 14px; }

  input, select {
    background: var(--secondary-bg); border: none; border-radius: 10px;
    padding: 12px; font-size: 15px; color: var(--text); width: 100%;
  }
  button.action {
    background: var(--button); color: var(--button-text); border: none;
    border-radius: 10px; padding: 12px 16px; font-size: 15px; font-weight: 600;
    cursor: pointer; white-space: nowrap;
  }
  .field { margin-bottom: 10px; }
  .field label { font-size: 13px; color: var(--hint); display: block; margin-bottom: 4px; }

  .back { color: var(--link); font-size: 15px; margin-bottom: 14px; cursor: pointer; }
  .detail-title { font-size: 24px; font-weight: 800; margin-bottom: 4px; }
  .detail-badge { display: inline-block; margin: 6px 0 16px; }
  .row { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid var(--secondary-bg); font-size: 14px; }
  .row .label { color: var(--hint); }
  .reasons { margin-top: 16px; }
  .reasons h3 { font-size: 15px; margin-bottom: 8px; }
  .reason-item { background: var(--secondary-bg); border-radius: 10px; padding: 10px 12px; margin-bottom: 6px; font-size: 13px; }
  .empty { color: var(--hint); text-align: center; padding: 40px 16px; font-size: 14px; }
  .error { color: var(--red); text-align: center; padding: 12px; font-size: 13px; }
  .disclaimer { color: var(--hint); font-size: 11px; text-align: center; padding: 16px; }
  .result-box { background: var(--secondary-bg); border-radius: 14px; padding: 16px; margin-top: 14px; }

  .tabbar {
    position: fixed; bottom: 0; left: 0; right: 0;
    display: flex; background: var(--bg); border-top: 1px solid var(--secondary-bg);
    padding-bottom: env(safe-area-inset-bottom, 0px);
    z-index: 10;
  }
  .tab {
    flex: 1; text-align: center; padding: 10px 4px 8px; font-size: 11px;
    color: var(--hint); cursor: pointer;
  }
  .tab .icon { font-size: 20px; display: block; margin-bottom: 2px; }
  .tab.active { color: var(--link); }

  .strategy-switch { display: flex; gap: 8px; padding: 0 16px 14px; }
  .strategy-pill {
    flex: 1; text-align: center; padding: 8px; border-radius: 10px;
    background: var(--secondary-bg); font-size: 13px; font-weight: 600;
    color: var(--hint); cursor: pointer;
  }
  .strategy-pill.active { background: var(--button); color: var(--button-text); }
 
  :root { --bg: #080d18; --text: #eef3ff; --hint: #8b98b6; --link: #7c9cff; --button: #637bff; --button-text: #ffffff; --secondary-bg: #121b2d; }
  body { background: radial-gradient(ellipse at 10% -10%, rgba(78,103,255,.22), transparent 38%), #080d18; color: var(--text); font-size: 14px; }
  header { padding: 22px 18px 18px; border: 0; background: linear-gradient(180deg, rgba(17,27,48,.96), rgba(8,13,24,.15)); }
  .brandline { display:flex; align-items:center; gap:12px; margin-bottom:18px; }
  .brandmark { display:grid; place-items:center; width:42px; height:42px; border-radius:14px; font-size:21px; font-weight:900; color:#fff; background:linear-gradient(135deg,#6d85ff,#43d6c5); box-shadow:0 8px 28px rgba(89,119,255,.25); }
  .brandtitle { font-size:18px; letter-spacing:1.5px; font-weight:900; }
  .brandtitle span { color:#7d91ff; }
  .sub { margin-top:3px; color:#91a0bf; }
  .strategy-pill { display:flex; align-items:center; justify-content:center; gap:8px; padding:10px 12px; border:1px solid rgba(116,142,255,.25); border-radius:12px; background:rgba(80,104,220,.13); color:#b8c4ff; font-size:11px; letter-spacing:1px; font-weight:800; }
  .paper-tag { border-radius:5px; padding:3px 6px; background:rgba(70,220,173,.13); color:#61e6bc; font-size:10px; letter-spacing:.7px; }
  .pulse-dot { width:7px; height:7px; border-radius:50%; background:#52e3b3; box-shadow:0 0 12px rgba(82,227,179,.8); }
  .hero { margin:0 12px 16px; padding:20px; border-radius:20px; border:1px solid rgba(125,145,255,.2); background:linear-gradient(135deg,rgba(29,43,78,.98),rgba(15,25,43,.96)); box-shadow:0 14px 34px rgba(0,0,0,.18); }
  .hero-kicker { color:#8fa2ff; font-size:10px; font-weight:800; letter-spacing:2px; margin-bottom:12px; }
  .hero-title { font-size:25px; line-height:1.15; font-weight:850; letter-spacing:-.6px; }
  .hero-copy { margin-top:10px; max-width:280px; color:#9eabc7; font-size:12px; line-height:1.5; }
  .hero-stats { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; margin-top:20px; }
  .hero-stats div { padding:10px 8px; border-radius:11px; background:rgba(255,255,255,.045); min-width:0; }
  .hero-stats strong { display:block; font-size:12px; color:#f2f5ff; white-space:nowrap; }
  .hero-stats span { display:block; margin-top:5px; color:#8e9bb8; font-size:10px; line-height:1.3; }
  .screen { padding:12px; }
  .add-row { margin:0 0 14px; gap:9px; }
  input, select { background:#111a2a; border:1px solid #202d45; border-radius:12px; color:#edf2ff; min-width:0; }
  input::placeholder { color:#687796; }
  button.action { border-radius:12px; background:linear-gradient(135deg,#7187ff,#5369e8); box-shadow:0 5px 18px rgba(83,105,232,.2); }
  .card, .result-box, .reason-item { background:linear-gradient(145deg,#141e31,#101827); border:1px solid #202d43; border-radius:16px; }
  .card { padding:15px; margin-bottom:10px; box-shadow:0 5px 15px rgba(0,0,0,.09); }
  .symbol { color:#f0f4ff; font-size:15px; letter-spacing:.1px; }
  .conf { color:#8492b0; line-height:1.45; }
  .badge { padding:7px 10px; font-size:11px; letter-spacing:.4px; }
  .LONG, .LONG_ALL { background:#123e39; color:#69efc0; border:1px solid #1e6b5c; }
  .SHORT, .SHORT_ALL { background:#482532; color:#ff9cb0; border:1px solid #783b4d; }
  .NEUTRAL, .MIXED { background:#252f42; color:#a8b4ca; border:1px solid #34415a; }
  .LONG_MAJORITY { background:#123e39; color:#69efc0; }
  .SHORT_MAJORITY { background:#482532; color:#ff9cb0; }
  .remove-btn { background:#202b40; color:#9ba9c3; }
  .field label, .row .label { color:#8c9ab7; }
  .row { border-color:#202b40; }
  .detail-title { letter-spacing:-.6px; }
  .reasons h3 { color:#dfe7ff; }
  .reason-item { color:#b9c5df; }
  .empty { color:#8997b4; padding:32px 16px; }
  .error { background:rgba(239,68,68,.09); border-radius:12px; }
  .disclaimer { color:#72809d; }
  .tabbar { background:rgba(9,15,27,.97); border-top:1px solid #1e2a40; backdrop-filter:blur(18px); }
  .tab { padding:10px 3px 9px; color:#7583a0; font-weight:600; }
  .tab .icon { font-size:19px; }
  .tab.active { color:#a7b5ff; }
  .tab.active .icon { filter:drop-shadow(0 0 7px rgba(116,142,255,.45)); }
  @media (min-width:600px) { .hero { max-width:720px; margin-left:auto; margin-right:auto; } .screen { max-width:744px; margin:0 auto; } header { padding-left:max(18px,calc((100% - 744px)/2)); padding-right:max(18px,calc((100% - 744px)/2)); } }
</style>
</head>
<body>
  <header>
    <div class="brandline"><span class="brandmark">S</span><div><div class="brandtitle">SIGNAL<span> / LAB</span></div><div class="sub">Крипторынок · аналитическая панель</div></div></div>
    <div class="strategy-pill active"><span class="pulse-dot"></span> ФИЛЬТР-ТРЕНД <span class="paper-tag">PAPER</span></div>
  </header>
  <section class="hero">
    <div class="hero-kicker">MARKET INTELLIGENCE</div>
    <div class="hero-title">Решения — по сигналам,<br>не по эмоциям.</div>
    <div class="hero-copy">Закрытые свечи · контроль риска · никаких реальных ордеров</div>
    <div class="hero-stats"><div><strong>5m / 15m</strong><span>Сканер Lighter</span></div><div><strong>3× max</strong><span>Лимит плеча</span></div><div><strong>0</strong><span>Автоордеров</span></div></div>
  </section>

  <!-- LIST -->
  <div class="screen active" id="screen-list">
    <div class="add-row">
      <input id="add-coin-input" placeholder="Например SOL/USDT">
      <button class="action" onclick="addCoin()">+</button>
    </div>
    <div id="error-list"></div>
    <div id="list"><div class="empty">Загрузка…</div></div>
  </div>

  <div class="screen" id="screen-detail">
    <div class="back" onclick="showScreen('list')">← Назад к списку</div>
    <div id="detail-content"></div>
  </div>

  <!-- RISK -->
  <div class="screen" id="screen-risk">
    <div class="field"><label>Пара</label><input id="risk-symbol" placeholder="BTC/USDT"></div>
    <div class="field"><label>Баланс, $</label><input id="risk-balance" type="number" placeholder="1000"></div>
    <div class="field"><label>Риск на сделку, %</label><input id="risk-pct" type="number" placeholder="2"></div>
    <button class="action" style="width:100%" onclick="calcRisk()">Посчитать</button>
    <div id="risk-result"></div>
  </div>

  <!-- MTF -->
  <div class="screen" id="screen-mtf">
    <div class="field"><label>Пара</label><input id="mtf-symbol" placeholder="BTC/USDT"></div>
    <button class="action" style="width:100%" onclick="calcMtf()">Сверить таймфреймы</button>
    <div id="mtf-result"></div>
  </div>

  <!-- BACKTEST -->
  <div class="screen" id="screen-backtest">
    <div class="field"><label>Пара</label><input id="bt-symbol" placeholder="BTC/USDT"></div>
    <div class="field"><label>Таймфрейм</label><input id="bt-timeframe" placeholder="1h" value="1h"></div>
    <button class="action" style="width:100%" onclick="runBacktest()">Запустить бэктест</button>
    <div id="bt-result"></div>
  </div>

  <!-- ALERTS -->
  <div class="screen" id="screen-alerts">
    <div class="field"><label>Пара</label><input id="alert-symbol" placeholder="BTC/USDT"></div>
    <div style="display:flex; gap:8px;">
      <select id="alert-op" style="flex:0 0 70px">
        <option value=">">&gt;</option>
        <option value="<">&lt;</option>
      </select>
      <input id="alert-price" type="number" placeholder="Цена">
    </div>
    <button class="action" style="width:100%; margin-top:10px" onclick="addAlert()">Поставить алерт</button>
    <div id="alerts-list" style="margin-top:16px"></div>
  </div>

  <div class="disclaimer">⚠️ Не финансовая рекомендация. Фьючерсы с плечом — высокий риск.</div>

  <div class="tabbar">
    <div class="tab active" data-screen="list" onclick="showScreen('list')"><span class="icon">📋</span>Список</div>
    <div class="tab" data-screen="risk" onclick="showScreen('risk')"><span class="icon">⚖️</span>Риск</div>
    <div class="tab" data-screen="mtf" onclick="showScreen('mtf')"><span class="icon">⏱</span>MTF</div>
    <div class="tab" data-screen="backtest" onclick="showScreen('backtest')"><span class="icon">📈</span>Бэктест</div>
    <div class="tab" data-screen="alerts" onclick="showScreen('alerts')"><span class="icon">🔔</span>Алерты</div>
  </div>

<script>
const tg = window.Telegram?.WebApp;
if (tg) { tg.ready(); tg.expand(); }

function authHeaders() {
  return { "Content-Type": "application/json", "X-Telegram-Init-Data": tg ? tg.initData : "" };
}

let signalsData = {};
const dirLabel = { LONG: "🟢 LONG", SHORT: "🔴 SHORT", NEUTRAL: "⚪ NEUTRAL" };
const riskLabel = { LOW: "низкий", MEDIUM: "средний", HIGH: "высокий" };

// ---------- navigation ----------
function showScreen(name) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById('screen-' + name).classList.add('active');
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.screen === name));
  if (tg) { name === 'detail' ? tg.BackButton.show() : tg.BackButton.hide(); }
  if (name === 'alerts') loadAlerts();
}
if (tg) tg.BackButton.onClick(() => showScreen('list'));

// ---------- list ----------
async function loadSignals() {
  try {
    const res = await fetch('/api/signals', { headers: authHeaders() });
    if (!res.ok) throw new Error('unauthorized');
    const data = await res.json();
    signalsData = {};
    data.forEach(s => signalsData[s.symbol] = s);
    renderList(data);
  } catch (e) {
    document.getElementById('list').innerHTML = '<div class="empty">Не удалось загрузить данные</div>';
  }
}

function renderList(items) {
  const list = document.getElementById('list');
  if (!items.length) {
    list.innerHTML = '<div class="empty">Пока нет данных — подожди немного</div>';
    return;
  }
  list.innerHTML = items.map(s => `
    <div class="card">
      <div style="flex:1" onclick="openDetail('${s.symbol}')">
        <div class="symbol">${s.symbol}</div>
        <div class="conf">Уверенность ${s.confidence}% · риск <span class="risk-${s.risk_level}">${riskLabel[s.risk_level]}</span></div>
      </div>
      <div class="badge ${s.direction}" onclick="openDetail('${s.symbol}')">${dirLabel[s.direction]}</div>
      <button class="remove-btn" onclick="removeCoin('${s.symbol}', event)">✕</button>
    </div>
  `).join('');
}

async function addCoin() {
  const input = document.getElementById('add-coin-input');
  const symbol = input.value.trim().toUpperCase();
  const errBox = document.getElementById('error-list');
  errBox.innerHTML = '';
  if (!symbol) return;
  try {
    const res = await fetch('/api/coins', { method: 'POST', headers: authHeaders(), body: JSON.stringify({ symbol }) });
    const data = await res.json();
    if (!res.ok) { errBox.innerHTML = `<div class="error">${data.error || 'ошибка'}</div>`; return; }
    input.value = '';
    loadSignals();
  } catch (e) { errBox.innerHTML = '<div class="error">Не удалось добавить</div>'; }
}

async function removeCoin(symbol, event) {
  event.stopPropagation();
  try {
    await fetch('/api/coins/' + encodeURIComponent(symbol), { method: 'DELETE', headers: authHeaders() });
    loadSignals();
  } catch (e) {}
}

function openDetail(symbol) {
  const s = signalsData[symbol];
  if (!s) return;
  document.getElementById('detail-content').innerHTML = `
    <div class="detail-title">${s.symbol}</div>
    <div class="badge detail-badge ${s.direction}">${dirLabel[s.direction]}</div>
    <div class="row"><span class="label">Уверенность</span><span>${s.confidence}%</span></div>
    <div class="row"><span class="label">Риск (волатильность)</span><span class="risk-${s.risk_level}">${riskLabel[s.risk_level]}</span></div>
    <div class="row"><span class="label">Цена</span><span>${s.entry}</span></div>
    <div class="row"><span class="label">Стоп-лосс (ориентир)</span><span>${s.stop_loss}</span></div>
    <div class="row"><span class="label">Тейк-профит (ориентир)</span><span>${s.take_profit}</span></div>
    <div class="row"><span class="label">Таймфрейм</span><span>${s.timeframe}</span></div>
    <div class="reasons"><h3>Почему такой расклад</h3>
      ${s.reasons.map(r => `<div class="reason-item">${r}</div>`).join('')}
    </div>`;
  showScreen('detail');
}

// ---------- risk calculator ----------
async function calcRisk() {
  const symbol = document.getElementById('risk-symbol').value.trim().toUpperCase();
  const balance = document.getElementById('risk-balance').value;
  const risk_pct = document.getElementById('risk-pct').value;
  const box = document.getElementById('risk-result');
  box.innerHTML = '<div class="empty">Считаю…</div>';
  try {
    const res = await fetch('/api/risk', { method: 'POST', headers: authHeaders(), body: JSON.stringify({ symbol, balance, risk_pct }) });
    const d = await res.json();
    if (!res.ok) { box.innerHTML = `<div class="error">${d.error}</div>`; return; }
    if (d.direction === 'NEUTRAL') { box.innerHTML = `<div class="result-box">${d.message}</div>`; return; }
    box.innerHTML = `
      <div class="result-box">
        <div class="badge ${d.direction}">${dirLabel[d.direction]}</div>
        <div class="row"><span class="label">Риск на сделку</span><span>$${d.risk_amount}</span></div>
        <div class="row"><span class="label">Цена входа</span><span>${d.entry}</span></div>
        <div class="row"><span class="label">Стоп-лосс</span><span>${d.stop_loss}</span></div>
        <div class="row"><span class="label">Размер позиции</span><span>${d.position_size} ${d.base_asset}</span></div>
        <div class="row"><span class="label">Объём в $</span><span>$${d.position_value}</span></div>
      </div>`;
  } catch (e) { box.innerHTML = '<div class="error">Не удалось посчитать</div>'; }
}

// ---------- MTF ----------
async function calcMtf() {
  const symbol = document.getElementById('mtf-symbol').value.trim().toUpperCase();
  const box = document.getElementById('mtf-result');
  box.innerHTML = '<div class="empty">Смотрю…</div>';
  try {
    const res = await fetch('/api/mtf/' + encodeURIComponent(symbol), { headers: authHeaders() });
    const d = await res.json();
    if (!res.ok) { box.innerHTML = `<div class="error">${d.error}</div>`; return; }
    const consensusText = {
      LONG_ALL: "🟢 Все таймфреймы за LONG", SHORT_ALL: "🔴 Все таймфреймы за SHORT",
      LONG_MAJORITY: "🟡 Перевес в LONG", SHORT_MAJORITY: "🟡 Перевес в SHORT", MIXED: "⚪ Расхождение"
    }[d.consensus];
    let rows = Object.entries(d.timeframes).map(([tf, s]) =>
      `<div class="row"><span class="label">${tf}</span><span>${dirLabel[s.direction]} (${s.confidence}%)</span></div>`
    ).join('');
    box.innerHTML = `<div class="result-box"><div style="margin-bottom:10px;font-weight:600">${consensusText}</div>${rows}</div>`;
  } catch (e) { box.innerHTML = '<div class="error">Не удалось получить данные</div>'; }
}

// ---------- backtest ----------
async function runBacktest() {
  const symbol = document.getElementById('bt-symbol').value.trim().toUpperCase();
  const timeframe = document.getElementById('bt-timeframe').value.trim() || '1h';
  const box = document.getElementById('bt-result');
  box.innerHTML = '<div class="empty">Считаю (может занять до минуты)…</div>';
  try {
    const res = await fetch(`/api/backtest/${encodeURIComponent(symbol)}?timeframe=${encodeURIComponent(timeframe)}`, { headers: authHeaders() });
    const d = await res.json();
    if (!res.ok) { box.innerHTML = `<div class="error">${d.error}</div>`; return; }
    const r = d.results;
    const wr = (dir) => r[dir].count ? (r[dir].wins / r[dir].count * 100).toFixed(1) : '0.0';
    box.innerHTML = `
      <div class="result-box">
        <div style="margin-bottom:10px">${d.symbol} · ${d.timeframe} · ${d.candles} свечей</div>
        <div class="row"><span class="label">LONG сигналов</span><span>${r.LONG.count}, winrate ${wr('LONG')}%</span></div>
        <div class="row"><span class="label">SHORT сигналов</span><span>${r.SHORT.count}, winrate ${wr('SHORT')}%</span></div>
      </div>
      <div class="disclaimer">Ретроспективная статистика по текущим правилам сигнала, не гарантия на будущее.</div>`;
  } catch (e) { box.innerHTML = '<div class="error">Не удалось посчитать</div>'; }
}

// ---------- alerts ----------
async function loadAlerts() {
  const box = document.getElementById('alerts-list');
  box.innerHTML = '<div class="empty">Загрузка…</div>';
  try {
    const res = await fetch('/api/alerts', { headers: authHeaders() });
    const d = await res.json();
    const entries = Object.entries(d);
    if (!entries.length) { box.innerHTML = '<div class="empty">Активных алертов нет</div>'; return; }
    box.innerHTML = entries.map(([id, a]) => `
      <div class="card">
        <div>${a.symbol} ${a.op} ${a.price}</div>
        <button class="remove-btn" onclick="removeAlert('${id}')">✕</button>
      </div>`).join('');
  } catch (e) { box.innerHTML = '<div class="error">Не удалось загрузить</div>'; }
}

async function addAlert() {
  const symbol = document.getElementById('alert-symbol').value.trim().toUpperCase();
  const op = document.getElementById('alert-op').value;
  const price = document.getElementById('alert-price').value;
  if (!symbol || !price) return;
  try {
    await fetch('/api/alerts', { method: 'POST', headers: authHeaders(), body: JSON.stringify({ symbol, op, price }) });
    document.getElementById('alert-symbol').value = '';
    document.getElementById('alert-price').value = '';
    loadAlerts();
  } catch (e) {}
}

async function removeAlert(id) {
  try {
    await fetch('/api/alerts/' + id, { method: 'DELETE', headers: authHeaders() });
    loadAlerts();
  } catch (e) {}
}

// Single strategy is intentionally fixed to guarded.

loadSignals();
setInterval(loadSignals, 60000);
</script>
</body>
</html>
"""


@app.route("/")
def panel():
    return render_template_string(PANEL_HTML)


@app.route("/api/signals")
@require_webapp_auth
def api_signals():
    return jsonify(list(SIGNALS_CACHE.values()))


@app.route("/api/coins", methods=["POST"])
@require_webapp_auth
def api_add_coin():
    data = request.get_json(force=True) or {}
    symbol = (data.get("symbol") or "").upper().strip()
    if not symbol or "/" not in symbol:
        return jsonify({"error": "Неверный формат. Пример: SOL/USDT"}), 400

    state = storage.load()
    symbols = get_panel_symbols(state)
    if symbol in symbols:
        return jsonify({"error": "Эта монета уже в панели"}), 400

    symbols.append(symbol)
    state["panel_symbols"] = symbols
    storage.save(state)

    try:
        sig = compute_signal_dict(symbol)
        if sig:
            SIGNALS_CACHE[symbol] = sig
    except Exception as e:
        return jsonify({"ok": True, "warning": f"добавлено, но анализ пока не удался: {e}"})

    return jsonify({"ok": True, "signal": SIGNALS_CACHE.get(symbol)})


@app.route("/api/coins/<path:symbol>", methods=["DELETE"])
@require_webapp_auth
def api_remove_coin(symbol):
    symbol = symbol.upper()
    state = storage.load()
    symbols = get_panel_symbols(state)
    state["panel_symbols"] = [s for s in symbols if s != symbol]
    storage.save(state)
    SIGNALS_CACHE.pop(symbol, None)
    return jsonify({"ok": True})


@app.route("/api/risk", methods=["POST"])
@require_webapp_auth
def api_risk():
    data = request.get_json(force=True) or {}
    symbol = (data.get("symbol") or "").upper().strip()
    try:
        balance = float(data.get("balance"))
        risk_pct = float(data.get("risk_pct"))
    except (TypeError, ValueError):
        return jsonify({"error": "Баланс и риск % должны быть числами"}), 400
    if not symbol or balance <= 0 or risk_pct <= 0:
        return jsonify({"error": "Проверь введённые значения"}), 400

    strategy = storage.load().get("strategy", "guarded")
    try:
        df = client.fetch_ohlcv(symbol, timeframe="1h", limit=260)
        if len(df) < 220:
            return jsonify({"error": "Недостаточно данных по этой паре"}), 400
        imbalance = client.fetch_order_book_imbalance(symbol)
        funding = client.fetch_funding_rate(symbol)
        signal = analyze(df, order_book_imbalance=imbalance, funding_rate=funding, strategy=strategy)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if signal.direction == "NEUTRAL":
        return jsonify({"direction": "NEUTRAL", "message": "Сигнал нейтральный — чёткого стопа для расчёта нет."})

    risk_amount = balance * risk_pct / 100
    stop_distance = abs(signal.entry - signal.stop_loss)
    position_size = risk_amount / stop_distance if stop_distance else 0
    position_value = position_size * signal.entry

    return jsonify({
        "direction": signal.direction,
        "risk_amount": round(risk_amount, 2),
        "entry": round(signal.entry, 4),
        "stop_loss": round(signal.stop_loss, 4),
        "position_size": round(position_size, 6),
        "position_value": round(position_value, 2),
        "base_asset": symbol.split("/")[0],
    })


@app.route("/api/mtf/<path:symbol>")
@require_webapp_auth
def api_mtf(symbol):
    symbol = symbol.upper()
    strategy = storage.load().get("strategy", "guarded")
    results = {}
    for tf in MTF_TIMEFRAMES:
        try:
            df = client.fetch_ohlcv(symbol, timeframe=tf, limit=260)
            if len(df) < 220:
                continue
            imbalance = client.fetch_order_book_imbalance(symbol)
            funding = client.fetch_funding_rate(symbol)
            sig = analyze(df, order_book_imbalance=imbalance, funding_rate=funding, strategy=strategy)
            results[tf] = {"direction": sig.direction, "confidence": sig.confidence}
        except Exception:
            pass

    if not results:
        return jsonify({"error": "Не удалось получить данные"}), 500

    longs = sum(1 for s in results.values() if s["direction"] == "LONG")
    shorts = sum(1 for s in results.values() if s["direction"] == "SHORT")
    total = len(results)
    if longs == total:
        consensus = "LONG_ALL"
    elif shorts == total:
        consensus = "SHORT_ALL"
    elif longs > shorts:
        consensus = "LONG_MAJORITY"
    elif shorts > longs:
        consensus = "SHORT_MAJORITY"
    else:
        consensus = "MIXED"

    return jsonify({"symbol": symbol, "timeframes": results, "consensus": consensus})


@app.route("/api/backtest/<path:symbol>")
@require_webapp_auth
def api_backtest(symbol):
    symbol = symbol.upper()
    timeframe = request.args.get("timeframe", "1h")
    try:
        limit = min(int(request.args.get("limit", 300)), 1000)
    except ValueError:
        limit = 300
    strategy = request.args.get("strategy") or storage.load().get("strategy", "guarded")
    if strategy not in STRATEGY_NAMES:
        strategy = "guarded"

    try:
        df = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if len(df) < 100:
            return jsonify({"error": "Недостаточно свечей для бэктеста"}), 400
        results = run_backtest(df, strategy=strategy)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "symbol": symbol, "timeframe": timeframe, "candles": len(df),
        "strategy": strategy, "results": results,
    })


@app.route("/api/strategy", methods=["GET", "POST"])
@require_webapp_auth
def api_strategy():
    state = storage.load()
    if state.get("strategy") != "guarded":
        state["strategy"] = "guarded"
        storage.save(state)
    return jsonify({
        "current": "guarded",
        "options": [{"key": "guarded", "name": "Фильтр-тренд"}],
    })


@app.route("/api/alerts", methods=["GET", "POST"])
@require_webapp_auth
def api_alerts():
    state = storage.load()
    chat_id = request.tg_user["id"]

    if request.method == "POST":
        data = request.get_json(force=True) or {}
        symbol = (data.get("symbol") or "").upper().strip()
        op = data.get("op")
        try:
            price = float(data.get("price"))
        except (TypeError, ValueError):
            return jsonify({"error": "Цена должна быть числом"}), 400
        if not symbol or op not in (">", "<"):
            return jsonify({"error": "Проверь пару и знак"}), 400

        alert_id = uuid.uuid4().hex[:8]
        state["price_alerts"][alert_id] = {"chat_id": chat_id, "symbol": symbol, "op": op, "price": price}
        storage.save(state)
        return jsonify({"ok": True, "id": alert_id})

    mine = {aid: a for aid, a in state["price_alerts"].items() if a["chat_id"] == chat_id}
    return jsonify(mine)


@app.route("/api/alerts/<alert_id>", methods=["DELETE"])
@require_webapp_auth
def api_delete_alert(alert_id):
    state = storage.load()
    chat_id = request.tg_user["id"]
    a = state["price_alerts"].get(alert_id)
    if not a or a["chat_id"] != chat_id:
        return jsonify({"error": "not found"}), 404
    state["price_alerts"].pop(alert_id, None)
    storage.save(state)
    return jsonify({"ok": True})


def run(port: int = None):
    port = port or int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
