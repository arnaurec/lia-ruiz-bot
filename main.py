import asyncio
import json
import logging
import os
import random
import time
import re
from collections import deque
from typing import Any, Deque, Dict, Tuple, Optional, List
from openai import OpenAI, RateLimitError, APIError
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, CommandHandler, filters
from telegram.error import BadRequest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("lia-bot")

# Environment Variables (igual)
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PUBLIC_URL = os.getenv("PUBLIC_URL")
PORT = int(os.getenv("PORT", "8080"))
OWNER_CHAT_ID = os.getenv("OWNER_CHAT_ID")

if not all([BOT_TOKEN, OPENAI_API_KEY, PUBLIC_URL]):
    raise RuntimeError("Faltan env vars: BOT_TOKEN, OPENAI_API_KEY, PUBLIC_URL")

client = OpenAI(api_key=OPENAI_API_KEY)

memory: Dict[str, Deque[Dict[str, str]]] = {}
user_last_message: Dict[str, float] = {}
user_message_count: Dict[str, int] = {}

MAX_HISTORY_PER_USER = 80
OPENAI_HISTORY_LIMIT = 20
MAX_MESSAGE_LENGTH = 1500
RATE_LIMIT_SECONDS = 0.5

# (Todo TYPO_PATTERNS, SLANG_REPLACEMENTS, FILLER_WORDS, etc. igual, omito por brevedad pero está intacto)

# SYSTEM_PROMPT, CLASSIFIER_PROMPT, GUARDIAN_PROMPT, funciones get_current_mood, apply_typos_and_slang, etc. → todo igual

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return
    user_text = msg.text.strip()
    if not user_text:
        return

    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    if not check_rate_limit(user_id):
        return

    conv_id, dm_topic_id = conv_id_and_topic(update)
    append_history(conv_id, "user", user_text)

    flags = classify(user_text)

    if flags.get("minor"):
        await alert_owner(context, f"🛑 Menor: {user_text[:100]}")
        return

    api_kwargs = {}
    if dm_topic_id is not None:
        api_kwargs["message_thread_id"] = dm_topic_id

    history = get_history(conv_id)
    raw_reply = generate_raw_reply(history, user_text, user_id)

    is_valid, reason = validate_human_tone(raw_reply) if raw_reply else (False, "none")

    if not is_valid:
        logger.warning(f"Tono no humano ({reason}): {raw_reply[:80]}...")
        await alert_owner(context, f"⚠️ Tono raro ({reason}): {raw_reply[:100]}")
        raw_reply = generate_raw_reply(history, user_text + " (contesta como si fueras tú, rápido, sin pensar)", user_id)
        is_valid, reason = validate_human_tone(raw_reply) if raw_reply else (False, "none")
        if not is_valid:
            raw_reply = random.choice(FALLBACK_RESPONSES)

    is_hot = flags.get("hot", False)
    part1, part2, typing_delay = process_reply_to_human(raw_reply, is_hot)

    append_history(conv_id, "assistant", part1)
    if part2:
        append_history(conv_id, "assistant", part2)

    # Typing protegido
    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing",
            **api_kwargs
        )
    except BadRequest as e:
        if "Chat actions can't be sent to channel direct messages chats" in str(e):
            logger.info(f"Ignorando typing en topic canal: {conv_id}")
        else:
            logger.warning(f"Error typing: {e}")
    except Exception as e:
        logger.warning(f"Error inesperado typing: {e}")

    await asyncio.sleep(typing_delay)

    # Envío con fix para topics: agregar reply_to_message_id si es topic
    send_kwargs = api_kwargs.copy()
    if dm_topic_id is not None and msg.message_id:
        send_kwargs["reply_to_message_id"] = msg.message_id  # Esto resuelve "topic must be specified" en muchos casos

    try:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=part1,
            **send_kwargs,
        )

        if part2:
            try:
                await context.bot.send_chat_action(
                    chat_id=update.effective_chat.id,
                    action="typing",
                    **api_kwargs
                )
            except BadRequest:
                pass

            await asyncio.sleep(random.uniform(2, 6))

            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=part2,
                **send_kwargs,  # mismo fix
            )

    except BadRequest as e:
        if "Channel direct messages topic must be specified" in str(e):
            logger.warning(f"Error topic must be specified - reintentando sin thread: {conv_id}")
            # Reintento sin message_thread_id (para casos raros donde Telegram confunde)
            try:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=part1,
                    reply_to_message_id=msg.message_id if msg else None
                )
                if part2:
                    await asyncio.sleep(random.uniform(2, 6))
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=part2,
                        reply_to_message_id=msg.message_id if msg else None
                    )
            except Exception as retry_e:
                logger.error(f"Reintento falló: {retry_e}")
        else:
            logger.error(f"Error enviando: {e}")
    except Exception as e:
        logger.error(f"Error enviando: {e}")

# (error_handler, main, if __name__ igual)

def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(error_handler)

    webhook_url = f"{PUBLIC_URL}/telegram/webhook"

    logger.info(f"Bot iniciado en {webhook_url}")

    app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path="telegram/webhook",
        webhook_url=webhook_url,
    )

if __name__ == "__main__":
    main()
