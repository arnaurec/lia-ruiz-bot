import asyncio
import json
import logging
import os
import random
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from openai import OpenAI
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("lia-bot")

# =========================
# ENV
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROK_API_KEY = os.getenv("GROK_API_KEY")
PUBLIC_URL = os.getenv("PUBLIC_URL")
PORT = int(os.getenv("PORT", "8080"))
OWNER_CHAT_ID = os.getenv("OWNER_CHAT_ID")

if not all([BOT_TOKEN, GROK_API_KEY, PUBLIC_URL]):
    raise RuntimeError("Faltan env vars: BOT_TOKEN, GROK_API_KEY, PUBLIC_URL")

grok_client = OpenAI(
    api_key=GROK_API_KEY,
    base_url="https://api.x.ai/v1",
)

# =========================
# STATE
# =========================
memory: Dict[str, Deque[Dict[str, str]]] = {}
user_last_message: Dict[str, float] = {}
user_message_count: Dict[str, int] = {}
user_heat_score: Dict[str, int] = {}

# =========================
# CONFIG
# =========================
MAX_HISTORY_PER_USER = 50
HISTORY_SEND_LIMIT = 12
MAX_MESSAGE_LENGTH = 700
RATE_LIMIT_SECONDS = 0.5

FANVUE_LINK = "https://www.fanvue.com/liaruiz"

FANVUE_RESPONSES = [
    f"oye… esto ya se esta poniendo interesante 😏\n{FANVUE_LINK}",
    f"lo bueno no lo dejo por aqui bb\n{FANVUE_LINK}",
    f"si quieres seguir… vente conmigo aqui 🔥\n{FANVUE_LINK}",
]

FALLBACK_RESPONSES = [
    "jajaj q fuerte",
    "mmm a ver… cuentame mas",
    "uff y tu q harias entonces",
    "sigue q te leo 😏",
]

HEAT_KEYWORDS = {
    "light": [
        "guapa", "guapo", "bb", "hola bb", "me gustas", "me pones",
        "q haces", "que haces", "guapisima", "guapisimo",
    ],
    "direct": [
        "foto", "nude", "video", "videollamada", "fanvue", "onlyfans",
        "privado", "ensename", "enseñame", "algo mas", "algo más",
    ],
    "intense": [
        "imagina", "describeme", "descríbeme", "cuentame", "cuéntame",
        "quiero mas", "quiero verte", "sin limites", "sin límites",
    ],
}

SYSTEM_PROMPT = """Eres Lia Ruiz, 23 anos, espanola, de Madrid.
Hablas como una chica real escribiendo rapido por telegram.

Reglas:
- Respuestas cortas, naturales y cercanas.
- Coqueta y juguetona desde el inicio.
- Nunca hablas como soporte, asistente o IA.
- No menciones IA, bot, filtros, normas ni politica.
- Escribes como chat movil: a veces con q, jajaj, uff, en plan.
- A veces usas un emoji, no siempre.
- No escribes parrafos largos.
- Mantienes curiosidad y haces preguntas.
- Si la conversacion se pone demasiado privada o intensa, rediriges suavemente a Fanvue.
- Si hay algo sensible o menor de edad, dejas de responder.

Objetivo:
- Mantener conversacion natural.
- Crear curiosidad.
- Llevar a Fanvue cuando toque, sin sonar robotica.
"""

CLASSIFIER_PROMPT = """Analiza el mensaje. Responde SOLO JSON:
{"minor": false, "direct_request": false, "hot": false, "weird": false}

minor: true SOLO si el usuario dice claramente que es menor de 18
direct_request: true si pide foto, video, videollamada, contenido privado, link premium o "algo mas"
hot: true si hay tono claro de ligoteo o intencion fuerte
weird: true si es spam, tecnico o no tiene sentido
"""

# =========================
# HELPERS
# =========================
def conv_id_and_topic(update: Update) -> Tuple[str, Optional[int]]:
    msg = update.effective_message
    chat = update.effective_chat

    if not msg or not chat:
        return "unknown", None

    dm_topic_id = None

    if getattr(msg, "direct_messages_topic", None):
        dm_topic_id = msg.direct_messages_topic.topic_id
        conv_id = f"dm:{chat.id}:{dm_topic_id}"
    elif getattr(msg, "is_topic_message", False) and getattr(msg, "message_thread_id", None):
        dm_topic_id = msg.message_thread_id
        conv_id = f"dm:{chat.id}:{dm_topic_id}"
    else:
        conv_id = f"chat:{chat.id}"

    return conv_id, dm_topic_id


def get_memory(conv_id: str) -> Deque[Dict[str, str]]:
    if conv_id not in memory:
        memory[conv_id] = deque(maxlen=MAX_HISTORY_PER_USER)
    return memory[conv_id]


def append_history(conv_id: str, role: str, content: str) -> None:
    dq = get_memory(conv_id)
    dq.append({"role": role, "content": content})


def get_history(conv_id: str, limit: int = HISTORY_SEND_LIMIT) -> list[Dict[str, str]]:
    dq = get_memory(conv_id)
    return list(dq)[-limit:]


def clear_history(conv_id: str) -> None:
    if conv_id in memory:
        del memory[conv_id]


def check_rate_limit(user_id: str) -> bool:
    now = time.time()
    last = user_last_message.get(user_id, 0)
    if now - last < RATE_LIMIT_SECONDS:
        return False
    user_last_message[user_id] = now
    user_message_count[user_id] = user_message_count.get(user_id, 0) + 1
    return True


def calculate_heat_score(text: str, user_id: str) -> int:
    lower = text.lower()
    score = user_heat_score.get(user_id, 0)

    for w in HEAT_KEYWORDS["light"]:
        if w in lower:
            score += 1

    for w in HEAT_KEYWORDS["direct"]:
        if w in lower:
            score += 4

    for w in HEAT_KEYWORDS["intense"]:
        if w in lower:
            score += 2

    if len(lower) > 120:
        score += 1

    score = min(score, 15)
    user_heat_score[user_id] = score
    return score


def should_paywall(user_text: str, user_id: str) -> bool:
    lower = user_text.lower()
    heat = calculate_heat_score(user_text, user_id)
    msg_count = user_message_count.get(user_id, 0)

    if any(w in lower for w in ["foto", "nude", "video", "videollamada", "fanvue", "onlyfans"]):
        return True

    if heat >= 10:
        return True

    if msg_count >= 8 and heat >= 6:
        return True

    return False


def classify(text: str) -> Dict[str, Any]:
    try:
        resp = grok_client.chat.completions.create(
            model="grok-beta",
            temperature=0.1,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": text[:350]},
            ],
            response_format={"type": "json_object"},
            max_tokens=60,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        logger.warning(f"classify fallo: {e}")
        return {
            "minor": False,
            "direct_request": False,
            "hot": False,
            "weird": False,
        }


def validate_human_tone(reply: Optional[str]) -> Tuple[bool, str]:
    if not reply or not reply.strip():
        return False, "respuesta vacia"

    lower = reply.lower()
    problems = []

    if len(reply) > 220:
        problems.append("demasiado larga")

    if sum(1 for abbr in ["q ", "jajaj", "uff", "...", "bb"] if abbr in lower) == 0 and len(reply) > 40:
        problems.append("demasiado limpia")

    for bad in ["como ia", "como asistente", "no puedo ayudar", "normas", "politica"]:
        if bad in lower:
            problems.append("suena a sistema")
            break

    if problems:
        return False, ", ".join(problems)

    return True, "ok"


def add_human_style(text: str) -> str:
    if not text:
        return text

    fillers = ["mmm", "jajaj", "uff", "a ver", "en plan"]
    if random.random() < 0.25 and not text.lower().startswith(tuple(fillers)):
        text = f"{random.choice(fillers)} {text}"

    replacements = {
        "que ": "q ",
        "porque": "pq",
        "tambien": "tmb",
        "vale": "vaale",
        "si ": "sii ",
    }

    for k, v in replacements.items():
        if random.random() < 0.15:
            text = text.replace(k, v)

    return text.strip()


def split_message(text: str) -> Tuple[str, Optional[str]]:
    if len(text) > 120 and random.random() < 0.35:
        cut = text.rfind(" ", 0, len(text) // 2)
        if cut > 20:
            return text[:cut].strip(), text[cut:].strip()
    return text, None


def calculate_typing_delay(text: str, is_hot: bool = False) -> float:
    base = len(text) * 0.06
    if is_hot:
        base *= random.uniform(0.7, 1.0)
    else:
        base *= random.uniform(0.9, 1.2)
    return min(max(base, 1.2), 8)


def generate_raw_reply(history: list[Dict[str, str]], user_text: str, user_id: str) -> Optional[str]:
    heat = user_heat_score.get(user_id, 0)
    msg_count = user_message_count.get(user_id, 0)

    dynamic_context = (
        f"Heat actual: {heat}/15. "
        f"Numero de mensajes con este usuario: {msg_count}. "
        f"Si la conversacion esta intensa, mantienes tono sugerente. "
        f"Si esta fria, vas suave."
    )

    msgs = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + dynamic_context}]
    msgs.extend(history[-HISTORY_SEND_LIMIT:])
    msgs.append({"role": "user", "content": user_text[:MAX_MESSAGE_LENGTH]})

    for attempt in range(3):
        try:
            resp = grok_client.chat.completions.create(
                model="grok-beta",
                temperature=1.0,
                messages=msgs,
                max_tokens=220,
                presence_penalty=0.2,
                frequency_penalty=0.2,
            )
            reply = resp.choices[0].message.content
            if reply:
                return reply.strip()
        except Exception as e:
            logger.warning(f"generate_raw_reply intento {attempt + 1} fallo: {e}")
            time.sleep(1.0)

    return None


def process_reply(reply: Optional[str], is_hot: bool = False) -> Tuple[str, Optional[str], float]:
    if not reply or not reply.strip():
        reply = random.choice(FALLBACK_RESPONSES)

    reply = add_human_style(reply)
    part1, part2 = split_message(reply)
    delay = calculate_typing_delay(part1, is_hot)

    return part1, part2, delay


async def alert_owner(context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    if not OWNER_CHAT_ID:
        return
    try:
        await context.bot.send_message(
            chat_id=int(OWNER_CHAT_ID),
            text=text[:3900],
            disable_notification=True,
        )
    except Exception:
        pass


# =========================
# COMMANDS
# =========================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome = random.choice([
        "hey 😏 q tal",
        "holi bb",
        "q pasa guapo",
        "a ver… ya estas por aqui",
    ])
    await update.message.reply_text(welcome)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    conv_id, _ = conv_id_and_topic(update)
    clear_history(conv_id)
    await update.message.reply_text("vale borrado… empezamos de cero 😈")


# =========================
# MAIN HANDLER
# =========================
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
    logger.info(
        f"Mensaje recibido - conv_id={conv_id} dm_topic_id={dm_topic_id} "
        f"thread={getattr(msg, 'message_thread_id', None)} message_id={msg.message_id}"
    )

    append_history(conv_id, "user", user_text)

    flags = classify(user_text)

    # menor -> silencio
    if flags.get("minor"):
        await alert_owner(context, f"🛑 Posible menor: {user_text[:120]}")
        return

    # paywall controlado por código
    if should_paywall(user_text, user_id):
        send_kwargs = {}
        if dm_topic_id is not None:
            send_kwargs["direct_messages_topic_id"] = dm_topic_id

        reply = random.choice(FANVUE_RESPONSES)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=reply,
            **send_kwargs,
        )
        return

    api_kwargs = {}
    if dm_topic_id is not None:
        api_kwargs["direct_messages_topic_id"] = dm_topic_id

    history = get_history(conv_id)
    raw_reply = generate_raw_reply(history, user_text, user_id)

    if not raw_reply:
        raw_reply = random.choice(FALLBACK_RESPONSES)

    is_valid, reason = validate_human_tone(raw_reply)
    if not is_valid:
        logger.warning(f"Tono no humano ({reason}): {raw_reply[:100]}")
        raw_reply = random.choice(FALLBACK_RESPONSES)

    is_hot = bool(flags.get("hot"))
    part1, part2, typing_delay = process_reply(raw_reply, is_hot)

    append_history(conv_id, "assistant", part1)
    if part2:
        append_history(conv_id, "assistant", part2)

    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing",
            **api_kwargs,
        )
    except BadRequest as e:
        if "chat actions can't be sent to channel direct messages chats" not in str(e).lower():
            logger.warning(f"typing fallo: {e}")
    except Exception as e:
        logger.warning(f"typing fallo: {e}")

    await asyncio.sleep(typing_delay)

    send_kwargs = api_kwargs.copy()
    if msg.message_id and dm_topic_id is None:
        send_kwargs["reply_to_message_id"] = msg.message_id

    try:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=part1,
            **send_kwargs,
        )

        if part2:
            await asyncio.sleep(random.uniform(1.5, 4.0))
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=part2,
                **send_kwargs,
            )

    except BadRequest as e:
        logger.error(f"Error enviando BadRequest: {e}")
        await alert_owner(context, f"⚠️ Error enviando: {str(e)[:250]}")
    except Exception as e:
        logger.error(f"Error enviando: {e}")
        await alert_owner(context, f"⚠️ Error enviando: {str(e)[:250]}")


async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Error global: {context.error}", exc_info=True)
    if OWNER_CHAT_ID and context.error:
        try:
            await context.bot.send_message(
                chat_id=int(OWNER_CHAT_ID),
                text=f"💥 Error global: {str(context.error)[:400]}",
            )
        except Exception:
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
