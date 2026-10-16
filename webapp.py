"""
webapp.py
Flask app serving a Telegram Mini App (WebApp): a list of tracked
assets with their current Long/Short/Neutral read, and a detail view
per asset with the reasoning, risk level, stop-loss/take-profit.

Runs in a background thread alongside the Telegram bot's polling loop
(started from bot.py). Read-only — serves cached analysis data, never
places trades or touches a wallet.
"""

import os
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

# Populated by bot.py's periodic refresh job: {symbol: signal_dict}
SIGNALS_CACHE: dict = {}

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
    --secondary-bg: var(--tg-theme-secondary-bg-color, #f0f0f0);
    --green: #22c55e;
    --red: #ef4444;
    --gray: #9ca3af;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding-top: env(safe-area-inset-top, 0px);
    padding-bottom: env(safe-area-inset-bottom, 0px);
  }
  header { padding: 16px; font-size: 20px; font-weight: 700; border-bottom: 1px solid var(--secondary-bg); }
  .sub { color: var(--hint); font-size: 13px; font-weight: 400; margin-top: 2px; }
  #list { padding: 8px 12px; }
  .card {
    display: flex; align-items: center; justify-content: space-between;
    background: var(--secondary-bg); border-radius: 14px; padding: 14px 16px;
    margin-bottom: 10px; cursor: pointer; transition: opacity .15s;
  }
  .card:active { opacity: 0.6; }
  .symbol { font-weight: 700; font-size: 16px; }
  .conf { color: var(--hint); font-size: 12px; margin-top: 2px; }
  .badge { padding: 6px 12px; border-radius: 999px; font-size: 13px; font-weight: 700; color: white; white-space: nowrap; }
  .LONG { background: var(--green); }
  .SHORT { background: var(--red); }
  .NEUTRAL { background: var(--gray); }
  .risk-LOW { color: var(--green); }
  .risk-MEDIUM { color: #eab308; }
  .risk-HIGH { color: var(--red); }
  #detail { display: none; padding: 16px; }
  #detail.active { display: block; }
  #list.hidden { display: none; }
  .back { color: var(--link); font-size: 15px; margin-bottom: 14px; cursor: pointer; }
  .detail-title { font-size: 24px; font-weight: 800; margin-bottom: 4px; }
  .detail-badge { display: inline-block; margin: 6px 0 16px; }
  .row { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid var(--secondary-bg); font-size: 14px; }
  .row .label { color: var(--hint); }
  .reasons { margin-top: 16px; }
  .reasons h3 { font-size: 15px; margin-bottom: 8px; }
  .reason-item { background: var(--secondary-bg); border-radius: 10px; padding: 10px 12px; margin-bottom: 6px; font-size: 13px; }
  .empty { color: var(--hint); text-align: center; padding: 40px 16px; font-size: 14px; }
  .disclaimer { color: var(--hint); font-size: 11px; text-align: center; padding: 16px; }
</style>
</head>
<body>
  <header>📊 Signal Panel<div class="sub">Технический расклад по активам</div></header>
  <div id="list"><div class="empty">Загрузка…</div></div>
  <div id="detail">
    <div class="back" onclick="showList()">← Назад к списку</div>
    <div id="detail-content"></div>
  </div>
  <div class="disclaimer">⚠️ Не финансовая рекомендация. Фьючерсы с плечом — высокий риск.</div>

<script>
const tg = window.Telegram?.WebApp;
if (tg) { tg.ready(); tg.expand(); }

let signalsData = {};
const dirLabel = { LONG: "🟢 LONG", SHORT: "🔴 SHORT", NEUTRAL: "⚪ NEUTRAL" };
const riskLabel = { LOW: "низкий", MEDIUM: "средний", HIGH: "высокий" };

async function loadSignals() {
  try {
    const res = await fetch('/api/signals');
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
    <div class="card" onclick="openDetail('${s.symbol}')">
      <div>
        <div class="symbol">${s.symbol}</div>
        <div class="conf">Уверенность ${s.confidence}% · риск <span class="risk-${s.risk_level}">${riskLabel[s.risk_level]}</span></div>
      </div>
      <div class="badge ${s.direction}">${dirLabel[s.direction]}</div>
    </div>
  `).join('');
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
  showDetail();
}

function showDetail() {
  document.getElementById('list').classList.add('hidden');
  document.getElementById('detail').classList.add('active');
  if (tg) tg.BackButton.show();
}
function showList() {
  document.getElementById('list').classList.remove('hidden');
  document.getElementById('detail').classList.remove('active');
  if (tg) tg.BackButton.hide();
}
if (tg) tg.BackButton.onClick(showList);

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
def api_signals():
    return jsonify(list(SIGNALS_CACHE.values()))


def run(port: int = None):
    port = port or int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
