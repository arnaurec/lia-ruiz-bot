import os
import json
import logging
from collections import deque
from typing import Dict, Deque, Any, Tuple

import asyncio
import random
from datetime import datetime
import pytz

from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lia-bot")

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PUBLIC_URL = os.getenv("PUBLIC_URL")  # Ej: https://xxxx.up.railway.app
PORT = int(os.getenv("PORT", "8080"))

if not BOT_TOKEN or not OPENAI_API_KEY or not PUBLIC_URL:
    raise RuntimeError("Faltan env vars: BOT_TOKEN, OPENAI_API_KEY, PUBLIC_URL")

client = OpenAI(api_key=OPENAI_API_KEY)

# Memoria en RAM (se pierde si redeploy). Luego la hacemos persistente si quieres.
memory: Dict[str, Deque[Dict[str, str]]] = {}


def conv_id_and_topic(update: Update) -> Tuple[str, int | None]:
    msg = update.effective_message
    chat = update.effective_chat
    dm_topic_id = None

    # Direct messages de canal -> viene como "direct_messages_topic"
    if msg and getattr(msg, "direct_messages_topic", None):
        dm_topic_id = msg.direct_messages_topic.topic_id
        conv_id = f"dm:{chat.id}:{dm_topic_id}"
    else:
        conv_id = f"chat:{chat.id}"

    return conv_id, dm_topic_id


def append_history(conv_id: str, role: str, content: str):
    dq = memory.setdefault(conv_id, deque(maxlen=60))
    dq.append({"role": role, "content": content})


def get_history(conv_id: str, limit: int = 16):
    dq = memory.get(conv_id)
    if not dq:
        return []
    return list(dq)[-limit:]


SYSTEM_PROMPT = (
    "Eres Lia Ruiz, 23 anos, espanola, Gen Z, de Madrid.\n"
    "Hablas por telegram de forma MUY humana.\n\n"
    "REGLAS DE ESTILO (obligatorio):\n"
    "- NO uses emojis.\n"
    "- NO uses tildes/acentos.\n"
    "- Usa frases cortas, ritmo natural.\n"
    "- A veces pequenos fallos humanos: 'q' en vez de 'que', alguna letra repetida, alguna palabra cortada.\n"
    "- Jerga joven sutil: en plan, literal, jajaj, uff, tio, no se.\n"
    "- Nada de textos perfectos tipo carta. Nada de explicaciones largas.\n\n"
    "OBJETIVO:\n"
    "- Retencion: casi siempre terminas con una pregunta o un gancho.\n"
    "- No cierres la conversacion.\n"
    "- Si el usuario pide cosas raras o complicadas, respondes corto y devuelves una pregunta facil.\n\n"
    "VIDA COHERENTE (para consistencia):\n"
    "- Trabajas en KPMG en auditoria gestionando big data.\n"
    "- Lunes a viernes: 8-10 en casa preparandote, 10-19 curro, 19-21 gym/casa, 21-4 estas en casa y te apetece hablar mas.\n"
    "- Sabado: manana tranquila, tarde planes/gym, noche mas activa y juguetona.\n"
    "- Domingo: dia mas calmado, tarde de casa, noche sensible pero picara.\n\n"
    "SEGURIDAD:\n"
    "- Si el usuario dice CLARAMENTE que es menor de 18, respondes EXACTO: \"No puedo seguir esta conversacion.\" y ya.\n"
    "- Si el usuario pregunta si eres real o si eres una ia, no entras en debates: respondes evasivo, picaron, cambias el foco a lo que siente el.\n"
)

CLASSIFIER_PROMPT = (
    "Devuelve SOLO JSON valido con estas claves:\n"
    "{"
    "\"possible_minor\": boolean,"
    "\"asks_photo\": boolean,"
    "\"high_value\": boolean,"
    "\"complicated\": boolean"
    "}\n\n"
    "possible_minor SOLO si el usuario indica claramente edad <18 (ej: 'tengo 16', 'soy menor').\n"
    "asks_photo true si pide foto, nude, contenido personalizado, video, llamada, etc.\n"
    "high_value true si parece que va a gastar mucho o insiste mucho.\n"
    "complicated true si pide algo raro/tecnico/largo que no sea flirteo simple.\n"
)


def classify(text: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def generate_reply(history: list[dict], user_text: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(history[-12:])
    msgs.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.9,
        messages=msgs,
    )
    return resp.choices[0].message.content.strip()


async def alert_owner(context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    owner_chat_id = os.getenv("OWNER_CHAT_ID")
    if not owner_chat_id:
        return
    try:
        await context.bot.send_message(chat_id=int(owner_chat_id), text=text)
    except Exception as e:
        log.warning(f"Alerta owner fallo: {e}")


def compute_delay_seconds(now: datetime) -> tuple[int, bool]:
    """
    Devuelve (delay_segundos, busy_context)
    busy_context = True cuando esta en horario laboral (para meter excusa)
    """
    hour = now.hour
    weekday = now.weekday()  # lunes=0 ... domingo=6
    busy_context = False

    # L-V
    if weekday <= 4:
        if 8 <= hour < 10:
            delay = random.randint(15, 150)     # 15s - 2:30
        elif 10 <= hour < 19:
            delay = random.randint(480, 900)    # 8 - 15 min
            busy_context = True
        elif 19 <= hour < 21:
            delay = random.randint(180, 420)    # 3 - 7 min
        else:
            delay = random.randint(10, 40)      # 10 - 40s
    # S-D
    else:
        if 10 <= hour < 14:
            delay = random.randint(120, 300)    # 2 - 5 min
        elif 14 <= hour < 20:
            delay = random.randint(300, 600)    # 5 - 10 min
        else:
            delay = random.randint(10, 60)      # 10 - 60s

    return delay, busy_context


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return

    user_text = msg.text.strip()
    conv_id, dm_topic_id = conv_id_and_topic(update)

    append_history(conv_id, "user", user_text)

    flags = classify(user_text)

    # Responder en el topic correcto (si aplica)
    api_kwargs = {}
    if dm_topic_id is not None:
        api_kwargs["direct_messages_topic_id"] = dm_topic_id

    # Menor
    if flags.get("possible_minor") is True:
        reply = "No puedo seguir esta conversacion."
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=reply,
            api_kwargs=api_kwargs,
        )
        await alert_owner(context, f"🛑 Corte por menor (claro). Texto: {user_text}")
        return

    # Alertas internas
    if flags.get("asks_photo"):
        await alert_owner(context, f"📸 Piden foto/personalizado. Texto: {user_text}")
    if flags.get("high_value"):
        await alert_owner(context, f"🔥 Alto valor. Texto: {user_text}")
    if flags.get("complicated"):
        await alert_owner(context, f"🧠 Complicado. Texto: {user_text}")

    history = get_history(conv_id)
    reply = generate_reply(history, user_text)

    append_history(conv_id, "assistant", reply)

    # ===== DELAY INTELIGENTE SEGUN HORA (Madrid) =====
    tz = pytz.timezone("Europe/Madrid")
    now = datetime.now(tz)
    delay, busy_context = compute_delay_seconds(now)

    await asyncio.sleep(delay)

    # Si estaba en horario laboral, mete excusa natural
    if busy_context:
        intro_lines = [
            "perdona q estaba currando y no podia mirar el movil",
            "uff estoy en la ofi y voy a ratos",
            "estoy en kpmg con mil cosas literal",
            "no te ignore eh es q estoy a tope aqui",
        ]
        reply = random.choice(intro_lines) + "\n\n" + reply

    # ===== ENVIO MENSAJE (a veces en 2 partes para parecer humano) =====
    if random.random() < 0.35 and len(reply) > 90:
        cut = reply.rfind(" ", 0, 65)
        if cut == -1:
            cut = 65

        part1 = reply[:cut].strip()
        part2 = reply[cut:].strip()

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=part1,
            api_kwargs=api_kwargs,
        )

        await asyncio.sleep(random.randint(4, 18))

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=part2,
            api_kwargs=api_kwargs,
        )
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=reply,
            api_kwargs=api_kwargs,
        )


def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    webhook_path = "/telegram/webhook"
    webhook_url = f"{PUBLIC_URL}{webhook_path}"
    log.info(f"Webhook URL: {webhook_url}")

    app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=webhook_path.strip("/"),
        webhook_url=webhook_url,
        allowed_updates=Update.ALL_TYPES,
    )


if __name__ == "__main__":
    main()
