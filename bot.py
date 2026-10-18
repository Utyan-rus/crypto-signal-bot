"""
bot.py
Telegram bot that analyzes crypto assets and suggests Long / Short /
Neutral based on technical indicators and risk level, plus a Mini App
panel, watchlists, price alerts, multi-timeframe checks and a backtest.

IMPORTANT:
- This bot does NOT place trades, does NOT hold funds or API trading
  keys, and does NOT connect to your wallet. It only reads public
  market data and sends you information in chat.
- Every suggestion is a rule-based heuristic, not financial advice.
  Crypto futures trading with leverage carries a high risk of losing
  more than your initial margin. You decide and execute trades
  yourself, with full responsibility for the outcome.
- Access is invite-only: a new chat must /start with the invite token
  (sent as a deep link) once before any other command works.

State lives in storage.py (a JSON file on a Railway volume) and is
always re-read fresh in each handler/job rather than cached in memory,
because the Mini App panel (webapp.py) can mutate the same file from
a different thread/request at any time.

Setup: see README.md
"""

import logging
import os
import threading
import uuid
from functools import wraps

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, WebAppInfo
from telegram.ext import Application, CommandHandler, ContextTypes

from exchange import ExchangeClient
from analysis import analyze, backtest
import storage
import webapp

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEFAULT_EXCHANGE = os.getenv("EXCHANGE_ID", "binance")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

client = ExchangeClient(DEFAULT_EXCHANGE)

CHECK_INTERVAL_SECONDS = 15 * 60       # /watch re-check interval
PANEL_REFRESH_SECONDS = 5 * 60         # /panel cache refresh interval
ALERT_CHECK_SECONDS = 60               # price alert check interval
MTF_TIMEFRAMES = ["15m", "1h", "4h"]   # used by /mtf

DEFAULT_PANEL_SYMBOLS = [s.strip().upper() for s in os.getenv(
    "PANEL_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT"
).split(",") if s.strip()]
PANEL_TIMEFRAME = os.getenv("PANEL_TIMEFRAME", "1h")

DISCLAIMER = (
    "\n\n⚠️ Это технический сигнал на основе индикаторов, а не финансовая "
    "рекомендация. Фьючерсы с плечом — высокий риск. Решение и ответственность — твои."
)


def panel_symbols(state: dict) -> list[str]:
    return state["panel_symbols"] if state.get("panel_symbols") else list(DEFAULT_PANEL_SYMBOLS)


# ---------------------------------------------------------------- auth ----

def require_auth(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        state = storage.load()
        if update.effective_chat.id not in state.get("authorized_chat_ids", []):
            await update.message.reply_text(
                "🔒 Этот бот доступен только по приглашению. "
                "Попроси ссылку-приглашение у владельца."
            )
            return
        return await func(update, context)
    return wrapper


WELCOME_TEXT = (
    "Привет! Я показываю технический расклад Long/Short по выбранной паре.\n\n"
    "/analyze BTC/USDT [таймфрейм] — разовый анализ\n"
    "/mtf BTC/USDT — сверка сигнала по 15m/1h/4h\n"
    "/backtest BTC/USDT [таймфрейм] [свечей] — ретроспективная статистика по сигналу\n"
    "/risk BTC/USDT баланс риск% — калькулятор размера позиции\n"
    "/watch BTC/USDT [таймфрейм] — авто-алерт при сильном сигнале\n"
    "/unwatch BTC/USDT, /mywatches\n"
    "/alert BTC/USDT > 70000 — ценовой алерт, /alerts, /unalert id\n"
    "/panel — панель со списком активов (там же калькулятор риска, MTF, бэктест, алерты)\n"
    "/addcoin BTC/USDT, /removecoin BTC/USDT, /coins — монеты в панели\n\n"
    "Я не торгую за тебя и не имею доступа к твоему кошельку."
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    state = storage.load()
    authorized = set(state.get("authorized_chat_ids", []))

    if chat_id in authorized:
        await update.message.reply_text(WELCOME_TEXT)
        return

    token = context.args[0] if context.args else None
    if token and token == state.get("invite_token"):
        authorized.add(chat_id)
        state["authorized_chat_ids"] = list(authorized)
        storage.save(state)
        await update.message.reply_text("Доступ открыт!\n\n" + WELCOME_TEXT)
    else:
        await update.message.reply_text(
            "🔒 Этот бот доступен только по приглашению. "
            "Перейди по ссылке-приглашению ещё раз или попроси новую у владельца."
        )


@require_auth
async def myinvite_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = storage.load()
    me = await context.bot.get_me()
    link = f"https://t.me/{me.username}?start={state['invite_token']}"
    await update.message.reply_text(f"Ссылка-приглашение:\n{link}")


# ------------------------------------------------------------ analysis ----

@require_auth
async def analyze_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Укажи пару, например: /analyze BTC/USDT")
        return

    symbol = context.args[0].upper()
    timeframe = context.args[1] if len(context.args) > 1 else "1h"

    await update.message.reply_text(f"Смотрю {symbol} ({timeframe})…")

    try:
        df = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
        if len(df) < 60:
            await update.message.reply_text("Недостаточно данных по этой паре/таймфрейму.")
            return

        imbalance = client.fetch_order_book_imbalance(symbol)
        funding = client.fetch_funding_rate(symbol)
        signal = analyze(df, order_book_imbalance=imbalance, funding_rate=funding)

        direction_emoji = {"LONG": "🟢 LONG", "SHORT": "🔴 SHORT", "NEUTRAL": "⚪ NEUTRAL"}
        risk_emoji = {"LOW": "🟢 низкий", "MEDIUM": "🟡 средний", "HIGH": "🔴 высокий"}
        reasons_text = "\n".join(f"• {r}" for r in signal.reasons)

        msg = (
            f"*{symbol}* ({timeframe})\n"
            f"Сигнал: *{direction_emoji[signal.direction]}*\n"
            f"Уверенность: {signal.confidence}%\n"
            f"Риск (волатильность): {risk_emoji[signal.risk_level]}\n\n"
            f"Цена: {signal.entry:.4f}\n"
            f"Стоп-лосс (ориентир): {signal.stop_loss:.4f}\n"
            f"Тейк-профит (ориентир): {signal.take_profit:.4f}\n\n"
            f"Почему:\n{reasons_text}"
            f"{DISCLAIMER}"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    except Exception as e:
        log.exception("Analyze failed")
        await update.message.reply_text(f"Не получилось получить данные: {e}")


@require_auth
async def mtf_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Укажи пару: /mtf BTC/USDT")
        return
    symbol = context.args[0].upper()
    await update.message.reply_text(f"Смотрю {symbol} на {', '.join(MTF_TIMEFRAMES)}…")

    results = {}
    for tf in MTF_TIMEFRAMES:
        try:
            df = client.fetch_ohlcv(symbol, timeframe=tf, limit=200)
            if len(df) < 60:
                continue
            imbalance = client.fetch_order_book_imbalance(symbol)
            funding = client.fetch_funding_rate(symbol)
            results[tf] = analyze(df, order_book_imbalance=imbalance, funding_rate=funding)
        except Exception:
            log.exception(f"mtf failed for {symbol} {tf}")

    if not results:
        await update.message.reply_text("Не удалось получить данные ни по одному таймфрейму.")
        return

    longs = sum(1 for s in results.values() if s.direction == "LONG")
    shorts = sum(1 for s in results.values() if s.direction == "SHORT")
    total = len(results)

    if longs == total:
        consensus = "🟢 Все таймфреймы за LONG — сильное согласие"
    elif shorts == total:
        consensus = "🔴 Все таймфреймы за SHORT — сильное согласие"
    elif longs > shorts:
        consensus = f"🟡 Перевес в LONG ({longs}/{total}), полного согласия нет"
    elif shorts > longs:
        consensus = f"🟡 Перевес в SHORT ({shorts}/{total}), полного согласия нет"
    else:
        consensus = "⚪ Таймфреймы расходятся — сигнал слабый"

    lines = [f"*{symbol}* — сверка по таймфреймам\n{consensus}\n"]
    emoji = {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "⚪"}
    for tf, s in results.items():
        lines.append(f"{tf}: {emoji[s.direction]} {s.direction} ({s.confidence}%)")
    lines.append(DISCLAIMER)

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


@require_auth
async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Использование: /backtest BTC/USDT [таймфрейм] [кол-во свечей]")
        return
    symbol = context.args[0].upper()
    timeframe = context.args[1] if len(context.args) > 1 else "1h"
    try:
        limit = min(int(context.args[2]), 1000) if len(context.args) > 2 else 500
    except ValueError:
        limit = 500

    await update.message.reply_text(f"Прогоняю бэктест по {symbol} ({timeframe}, {limit} свечей)…")

    try:
        df = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if len(df) < 100:
            await update.message.reply_text("Маловато данных для бэктеста.")
            return

        results = backtest(df)
        lines = [f"*{symbol}* бэктест ({timeframe}, {len(df)} свечей, горизонт 24 свечи)\n"]
        for direction in ("LONG", "SHORT"):
            c = results[direction]["count"]
            w = results[direction]["wins"]
            wr = (w / c * 100) if c else 0
            lines.append(f"{direction}: {c} сигналов, winrate {wr:.1f}%")
        lines.append(
            "\n⚠️ Это ретроспективная статистика по текущим правилам расчёта сигнала "
            "на исторических данных — не гарантия на будущее."
        )
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    except Exception as e:
        log.exception("Backtest failed")
        await update.message.reply_text(f"Не получилось посчитать бэктест: {e}")


@require_auth
async def risk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 3:
        await update.message.reply_text(
            "Использование: /risk BTC/USDT 1000 2\n(пара, баланс в $, риск в % от баланса)"
        )
        return

    symbol = context.args[0].upper()
    try:
        balance = float(context.args[1])
        risk_pct = float(context.args[2])
    except ValueError:
        await update.message.reply_text("Баланс и риск % должны быть числами.")
        return

    if balance <= 0 or risk_pct <= 0:
        await update.message.reply_text("Баланс и риск % должны быть больше нуля.")
        return

    try:
        df = client.fetch_ohlcv(symbol, timeframe="1h", limit=200)
        if len(df) < 60:
            await update.message.reply_text("Недостаточно данных по этой паре.")
            return
        imbalance = client.fetch_order_book_imbalance(symbol)
        funding = client.fetch_funding_rate(symbol)
        signal = analyze(df, order_book_imbalance=imbalance, funding_rate=funding)

        if signal.direction == "NEUTRAL":
            await update.message.reply_text(
                f"{symbol}: сигнал сейчас нейтральный, чёткого сетапа (и стопа) для расчёта нет."
            )
            return

        risk_amount = balance * risk_pct / 100
        stop_distance = abs(signal.entry - signal.stop_loss)
        if stop_distance == 0:
            await update.message.reply_text("Не удалось посчитать дистанцию до стопа.")
            return

        position_size = risk_amount / stop_distance
        position_value = position_size * signal.entry
        base_asset = symbol.split("/")[0]

        msg = (
            f"*{symbol}* — расчёт размера позиции ({signal.direction})\n\n"
            f"Риск на сделку: ${risk_amount:.2f} ({risk_pct}% от ${balance:.2f})\n"
            f"Цена входа: {signal.entry:.4f}\n"
            f"Стоп-лосс (ориентир): {signal.stop_loss:.4f} (дистанция {stop_distance:.4f})\n\n"
            f"Размер позиции: {position_size:.6f} {base_asset}\n"
            f"Объём в $: ${position_value:.2f}\n\n"
            f"Это математика под твой риск на сделку, не рекомендация по плечу — "
            f"сколько плеча использовать, решаешь сам."
            f"{DISCLAIMER}"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    except Exception as e:
        log.exception("Risk calc failed")
        await update.message.reply_text(f"Не получилось посчитать: {e}")


# ------------------------------------------------------------ watchlist ----

@require_auth
async def watch_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Укажи пару, например: /watch BTC/USDT")
        return

    symbol = context.args[0].upper()
    timeframe = context.args[1] if len(context.args) > 1 else "1h"
    chat_id = str(update.effective_chat.id)

    state = storage.load()
    state["watchlist"][chat_id] = {"symbol": symbol, "timeframe": timeframe, "min_confidence": 60}
    storage.save(state)

    await update.message.reply_text(
        f"Буду проверять {symbol} ({timeframe}) каждые {CHECK_INTERVAL_SECONDS // 60} мин "
        f"и напишу, когда уверенность сигнала ≥60%.\nОстановить: /unwatch {symbol}"
    )


@require_auth
async def unwatch_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    state = storage.load()
    w = state["watchlist"].pop(chat_id, None)
    storage.save(state)
    if w:
        await update.message.reply_text(f"Слежение за {w['symbol']} остановлено.")
    else:
        await update.message.reply_text("У тебя сейчас ничего не отслеживается.")


@require_auth
async def mywatches_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = storage.load()
    w = state["watchlist"].get(str(update.effective_chat.id))
    if w:
        await update.message.reply_text(f"Отслеживается: {w['symbol']} ({w['timeframe']})")
    else:
        await update.message.reply_text("Список пуст. Добавь парой через /watch BTC/USDT")


async def check_watchlist(context: ContextTypes.DEFAULT_TYPE):
    """Runs periodically; pushes a message only when a watched pair has a strong signal."""
    state = storage.load()
    for chat_id_str, w in list(state["watchlist"].items()):
        try:
            df = client.fetch_ohlcv(w["symbol"], timeframe=w["timeframe"], limit=200)
            if len(df) < 60:
                continue
            imbalance = client.fetch_order_book_imbalance(w["symbol"])
            funding = client.fetch_funding_rate(w["symbol"])
            signal = analyze(df, order_book_imbalance=imbalance, funding_rate=funding)

            if signal.direction != "NEUTRAL" and signal.confidence >= w["min_confidence"]:
                direction_emoji = {"LONG": "🟢 LONG", "SHORT": "🔴 SHORT"}
                msg = (
                    f"🔔 Автосигнал по *{w['symbol']}* ({w['timeframe']})\n"
                    f"{direction_emoji[signal.direction]} — уверенность {signal.confidence}%\n"
                    f"Цена: {signal.entry:.4f} | Стоп: {signal.stop_loss:.4f} | "
                    f"Тейк: {signal.take_profit:.4f}"
                    f"{DISCLAIMER}"
                )
                await context.bot.send_message(chat_id=int(chat_id_str), text=msg, parse_mode="Markdown")
        except Exception:
            log.exception(f"Watch check failed for chat {chat_id_str} ({w['symbol']})")


# -------------------------------------------------------------- alerts ----

@require_auth
async def alert_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 3 or args[1] not in (">", "<"):
        await update.message.reply_text("Использование: /alert BTC/USDT > 70000")
        return

    symbol = args[0].upper()
    op = args[1]
    try:
        price = float(args[2])
    except ValueError:
        await update.message.reply_text("Цена должна быть числом.")
        return

    state = storage.load()
    alert_id = uuid.uuid4().hex[:8]
    state["price_alerts"][alert_id] = {
        "chat_id": update.effective_chat.id, "symbol": symbol, "op": op, "price": price
    }
    storage.save(state)
    await update.message.reply_text(f"Алерт поставлен: {symbol} {op} {price}\nid: {alert_id} (для /unalert)")


@require_auth
async def unalert_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Укажи id алерта: /unalert <id> (см. /alerts)")
        return
    alert_id = context.args[0]
    state = storage.load()
    if state["price_alerts"].pop(alert_id, None):
        storage.save(state)
        await update.message.reply_text("Алерт удалён.")
    else:
        await update.message.reply_text("Такого алерта нет.")


@require_auth
async def alerts_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    state = storage.load()
    mine = {aid: a for aid, a in state["price_alerts"].items() if a["chat_id"] == chat_id}
    if not mine:
        await update.message.reply_text("Активных алертов нет.")
        return
    lines = [f"{aid}: {a['symbol']} {a['op']} {a['price']}" for aid, a in mine.items()]
    await update.message.reply_text("\n".join(lines))


async def check_price_alerts(context: ContextTypes.DEFAULT_TYPE):
    state = storage.load()
    to_remove = []
    for alert_id, a in list(state["price_alerts"].items()):
        try:
            last = client.fetch_last_price(a["symbol"])
            triggered = (a["op"] == ">" and last > a["price"]) or (a["op"] == "<" and last < a["price"])
            if triggered:
                await context.bot.send_message(
                    chat_id=a["chat_id"],
                    text=f"🔔 {a['symbol']} {a['op']} {a['price']} — сейчас {last:.4f}",
                )
                to_remove.append(alert_id)
        except Exception:
            log.exception(f"Alert check failed for {alert_id}")

    if to_remove:
        state = storage.load()
        for aid in to_remove:
            state["price_alerts"].pop(aid, None)
        storage.save(state)


# --------------------------------------------------------------- panel ----

async def refresh_one_symbol(symbol: str):
    try:
        df = client.fetch_ohlcv(symbol, timeframe=PANEL_TIMEFRAME, limit=200)
        if len(df) < 60:
            return
        imbalance = client.fetch_order_book_imbalance(symbol)
        funding = client.fetch_funding_rate(symbol)
        signal = analyze(df, order_book_imbalance=imbalance, funding_rate=funding)
        webapp.SIGNALS_CACHE[symbol] = {
            "symbol": symbol,
            "timeframe": PANEL_TIMEFRAME,
            "direction": signal.direction,
            "confidence": signal.confidence,
            "risk_level": signal.risk_level,
            "entry": round(signal.entry, 4),
            "stop_loss": round(signal.stop_loss, 4),
            "take_profit": round(signal.take_profit, 4),
            "reasons": signal.reasons,
        }
    except Exception:
        log.exception(f"Panel refresh failed for {symbol}")


async def refresh_panel_cache(context: ContextTypes.DEFAULT_TYPE):
    state = storage.load()
    symbols = panel_symbols(state)
    for symbol in symbols:
        await refresh_one_symbol(symbol)
    # drop cached entries for coins removed (e.g. via the panel UI) since the last refresh
    for cached_symbol in list(webapp.SIGNALS_CACHE.keys()):
        if cached_symbol not in symbols:
            webapp.SIGNALS_CACHE.pop(cached_symbol, None)


@require_auth
async def panel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    public_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN") or os.getenv("PANEL_PUBLIC_DOMAIN")
    panel_url = os.getenv("PANEL_URL") or (f"https://{public_domain}" if public_domain else None)

    if not panel_url:
        await update.message.reply_text(
            "Панель ещё не настроена: у сервиса нет публичного домена. "
            "Сгенерируй домен на Railway и задай переменную PANEL_URL или RAILWAY_PUBLIC_DOMAIN."
        )
        return

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 Открыть панель", web_app=WebAppInfo(url=panel_url))]
    ])
    await update.message.reply_text(
        f"Список активов, калькулятор риска, MTF, бэктест и алерты — всё внутри. "
        f"Список обновляется каждые {PANEL_REFRESH_SECONDS // 60} мин.",
        reply_markup=keyboard,
    )


@require_auth
async def addcoin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Укажи пару: /addcoin SOL/USDT")
        return
    symbol = context.args[0].upper()
    state = storage.load()
    symbols = panel_symbols(state)
    if symbol in symbols:
        await update.message.reply_text(f"{symbol} уже в панели.")
        return

    symbols = symbols + [symbol]
    state["panel_symbols"] = symbols
    storage.save(state)

    await update.message.reply_text(f"{symbol} добавлен. Считаю сигнал…")
    await refresh_one_symbol(symbol)
    await update.message.reply_text(f"Готово — {symbol} теперь в /panel.")


@require_auth
async def removecoin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Укажи пару: /removecoin SOL/USDT")
        return
    symbol = context.args[0].upper()
    state = storage.load()
    symbols = panel_symbols(state)
    if symbol not in symbols:
        await update.message.reply_text(f"{symbol} и так нет в панели.")
        return

    symbols = [s for s in symbols if s != symbol]
    state["panel_symbols"] = symbols
    storage.save(state)
    webapp.SIGNALS_CACHE.pop(symbol, None)

    await update.message.reply_text(f"{symbol} убран из панели.")


@require_auth
async def coins_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = storage.load()
    await update.message.reply_text("Монеты в панели:\n" + "\n".join(panel_symbols(state)))


def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан. Создай .env файл (см. .env.example).")

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("myinvite", myinvite_cmd))
    app.add_handler(CommandHandler("analyze", analyze_cmd))
    app.add_handler(CommandHandler("mtf", mtf_cmd))
    app.add_handler(CommandHandler("backtest", backtest_cmd))
    app.add_handler(CommandHandler("risk", risk_cmd))
    app.add_handler(CommandHandler("watch", watch_cmd))
    app.add_handler(CommandHandler("unwatch", unwatch_cmd))
    app.add_handler(CommandHandler("mywatches", mywatches_cmd))
    app.add_handler(CommandHandler("alert", alert_cmd))
    app.add_handler(CommandHandler("unalert", unalert_cmd))
    app.add_handler(CommandHandler("alerts", alerts_cmd))
    app.add_handler(CommandHandler("panel", panel_cmd))
    app.add_handler(CommandHandler("addcoin", addcoin_cmd))
    app.add_handler(CommandHandler("removecoin", removecoin_cmd))
    app.add_handler(CommandHandler("coins", coins_cmd))

    if app.job_queue:
        app.job_queue.run_repeating(check_watchlist, interval=CHECK_INTERVAL_SECONDS, first=30)
        app.job_queue.run_repeating(refresh_panel_cache, interval=PANEL_REFRESH_SECONDS, first=5)
        app.job_queue.run_repeating(check_price_alerts, interval=ALERT_CHECK_SECONDS, first=15)

    threading.Thread(target=webapp.run, daemon=True).start()
    log.info("Bot started, webapp thread launched")
    app.run_polling()


if __name__ == "__main__":
    main()
