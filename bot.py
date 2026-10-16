"""
bot.py
Telegram bot that analyzes a chosen crypto asset and suggests
Long / Short / Neutral based on technical indicators and risk level.

IMPORTANT:
- This bot does NOT place trades, does NOT hold funds or API trading
  keys, and does NOT connect to your wallet. It only reads public
  market data and sends you a suggestion in chat.
- The suggestion is a rule-based heuristic, not financial advice.
  Crypto futures trading with leverage carries a high risk of losing
  more than your initial margin. You decide and execute trades
  yourself, with full responsibility for the outcome.

Setup:
  1. pip install -r requirements.txt
  2. Create a bot with @BotFather on Telegram, get the token.
  3. Put the token in a .env file: TELEGRAM_BOT_TOKEN=xxxx
  4. Run: python bot.py

Usage in Telegram:
  /analyze BTC/USDT
  /analyze ETH/USDT 4h       (optional timeframe, default 1h)
"""

import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from exchange import ExchangeClient
from analysis import analyze

# chat_id -> {"symbol": str, "timeframe": str, "min_confidence": int}
WATCHLIST: dict[int, dict] = {}
CHECK_INTERVAL_SECONDS = 15 * 60  # how often /watch re-checks each pair

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEFAULT_EXCHANGE = os.getenv("EXCHANGE_ID", "binance")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

client = ExchangeClient(DEFAULT_EXCHANGE)

DISCLAIMER = (
    "\n\n⚠️ Это технический сигнал на основе индикаторов, а не финансовая "
    "рекомендация. Фьючерсы с плечом — высокий риск. Решение и ответственность — твои."
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я показываю технический расклад Long/Short по выбранной паре.\n\n"
        "/analyze BTC/USDT [таймфрейм] — разовый анализ\n"
        "/watch BTC/USDT [таймфрейм] — буду сам писать, когда появится сильный сигнал\n"
        "/unwatch BTC/USDT — остановить слежение за парой\n"
        "/mywatches — список того, что отслеживается\n\n"
        "Я не торгую за тебя и не имею доступа к твоему кошельку."
    )


async def watch_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Укажи пару, например: /watch BTC/USDT")
        return

    symbol = context.args[0].upper()
    timeframe = context.args[1] if len(context.args) > 1 else "1h"
    chat_id = update.effective_chat.id

    WATCHLIST[chat_id] = {"symbol": symbol, "timeframe": timeframe, "min_confidence": 60}

    await update.message.reply_text(
        f"Буду проверять {symbol} ({timeframe}) каждые {CHECK_INTERVAL_SECONDS // 60} мин "
        f"и напишу, когда уверенность сигнала ≥60%.\n"
        f"Остановить: /unwatch {symbol}"
    )


async def unwatch_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in WATCHLIST:
        symbol = WATCHLIST.pop(chat_id)["symbol"]
        await update.message.reply_text(f"Слежение за {symbol} остановлено.")
    else:
        await update.message.reply_text("У тебя сейчас ничего не отслеживается.")


async def mywatches_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    w = WATCHLIST.get(chat_id)
    if w:
        await update.message.reply_text(f"Отслеживается: {w['symbol']} ({w['timeframe']})")
    else:
        await update.message.reply_text("Список пуст. Добавь парой через /watch BTC/USDT")


async def check_watchlist(context: ContextTypes.DEFAULT_TYPE):
    """Runs periodically; pushes a message only when a watched pair has a strong signal."""
    for chat_id, w in list(WATCHLIST.items()):
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
                await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
        except Exception:
            log.exception(f"Watch check failed for chat {chat_id} ({w['symbol']})")


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


def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан. Создай .env файл (см. .env.example).")

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze_cmd))
    app.add_handler(CommandHandler("watch", watch_cmd))
    app.add_handler(CommandHandler("unwatch", unwatch_cmd))
    app.add_handler(CommandHandler("mywatches", mywatches_cmd))

    if app.job_queue:
        app.job_queue.run_repeating(check_watchlist, interval=CHECK_INTERVAL_SECONDS, first=30)

    log.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
