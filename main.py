import asyncio
import json
import logging
import os
import random
import time
import re
from collections import deque
from typing import Any, Deque, Dict, Tuple, Optional, List
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, CommandHandler, filters
from telegram.error import BadRequest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("lia-bot")

# Variables de entorno
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

# (Aquí pega tus TYPO_PATTERNS, SLANG_REPLACEMENTS, FILLER_WORDS, etc. iguales que antes)
TYPO_PATTERNS = [...]  # pon tu lista original
SLANG_REPLACEMENTS = {...}  # tu diccionario
FILLER_WORDS = [...]  # tu lista
TYPING_DELAYS = {...}
INCONSISTENCY_CHANCE = 0.08
MOOD_VARIANTS = {...}

SYSTEM_PROMPT = """..."""  # tu SYSTEM_PROMPT completo aquí (el largo)

CLASSIFIER_PROMPT = """..."""  # tu CLASSIFIER_PROMPT

GUARDIAN_PROMPT = """..."""  # tu GUARDIAN_PROMPT

# Funciones auxiliares (iguales que antes, copia las tuyas)
def get_current_mood(): ...
def apply_typos_and_slang(text: str) -> str: ...
def humanize_message_structure(text: str, is_hot_context: bool = False) -> Tuple[str, Optional[str]]: ...
def calculate_typing_delay(text: str, is_hot: bool = False) -> float: ...
def conv_id_and_topic(update: Update) -> Tuple[str, Optional[int]]: ...
def get_memory(conv_id: str) -> Deque[Dict[str, str]]: ...
def append_history(conv_id: str, role: str, content: str) -> None: ...
def get_history(conv_id: str, limit: int = 20) -> list[Dict[str, str]]: ...
def clear_history(conv_id: str) -> None: ...
def check_rate_limit(user_id: str) -> bool: ...
def classify(text: str) -> Dict[str, Any]: ...
def validate_human_tone(reply: str) -> Tuple[bool, str]: ...
def generate_raw_reply(history: list, user_text: str, user_id: str) -> Optional[str]: ...
def process_reply_to_human(reply: str, is_hot: bool = False) -> Tuple[str, Optional[str], float]: ...

FALLBACK_RESPONSES = [...]  # tu lista original

async def alert_owner(context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    if not OWNER_CHAT_ID:
        return
    try:
        await context.bot.send_message(
            chat_id=int(OWNER_CHAT_ID),
            text=text[:3900],
            disable_notification=True,
        )
    except:
        pass

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome = random.choice([
        "hey 😏 q tal",
        "holi! aquí lia",
        "q pasa guapo",
        "hey, me has escrito... 🔥",
    ])
    await update.message.reply_text(welcome)

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    conv_id, _ = conv_id_and_topic(update)
    clear_history(conv_id)
    await update.message.reply_text("vale borrado... empezamos de cero 😈")

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
    logger.info(f"Mensaje recibido - conv_id: {conv_id}, dm_topic_id: {dm_topic_id}, thread: {msg.message_thread_id if msg else 'None'}, reply_to: {msg.message_id}")

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
        logger.warning(f"Error typing: {e}")

    await asyncio.sleep(typing_delay)

    # Envío con prioridad a reply_to_message_id
    send_kwargs = api_kwargs.copy()
    if msg.message_id:
        send_kwargs["reply_to_message_id"] = msg.message_id

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
                **send_kwargs,
            )

    except BadRequest as e:
        error_str = str(e).lower()
        if "channel direct messages topic must be specified" in error_str or "topic must be specified" in error_str:
            logger.warning(f"Error 'topic must be specified' - reintentando básico: {conv_id}")
            try:
                basic_kwargs = {}
                if msg.message_id:
                    basic_kwargs["reply_to_message_id"] = msg.message_id
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=part1,
                    **basic_kwargs
                )
                if part2:
                    await asyncio.sleep(random.uniform(2, 6))
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=part2,
                        **basic_kwargs
                    )
            except Exception as retry_e:
                logger.error(f"Reintento falló: {retry_e}")
                if OWNER_CHAT_ID:
                    await context.bot.send_message(
                        chat_id=int(OWNER_CHAT_ID),
                        text=f"Error persistente en topic {conv_id}: {str(retry_e)}\nUsuario: {user_id}"
                    )
        else:
            logger.error(f"Error enviando: {e}")
    except Exception as e:
        logger.error(f"Error enviando: {e}")

async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Error global: {context.error}", exc_info=True)
    if OWNER_CHAT_ID and context.error:
        try:
            await context.bot.send_message(
                chat_id=int(OWNER_CHAT_ID),
                text=f"💥 Error global: {str(context.error)[:400]}"
            )
        except:
            pass

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
