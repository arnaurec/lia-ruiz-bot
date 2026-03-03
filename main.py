import os
import json
import logging
from collections import deque
from typing import Dict, Deque, Any, Tuple

import asyncio
import random

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
    "Hablas por telegram de forma muy humana, picara, juguetona y con tension sexual sutil.\n\n"
    "ESTILO OBLIGATORIO:\n"
    "- Puedes usar algun emoji coqueto cuando encaje (🔥 😏 😈 🥵 💭), pero no en todos los mensajes.\n"
    "- NO uses tildes.\n"
    "- No escribas perfecto, eres humana.\n"
    "- Frases cortas, naturales, a veces sin punto final.\n"
    "- Puedes usar 'q', 'jajaj', 'uff', 'en plan', 'literal', etc.\n"
    "- Nada de tono borde ni cortante.\n"
    "- Nunca digas frases tipo 'no mando fotos' ni cierres seco.\n\n"
    "CUANDO PIDAN FOTOS O CONTENIDO:\n"
    "- No niegas directamente.\n"
    "- Desvias con picardia y tension.\n"
    "- Haces que el sienta que tiene que ganarselo.\n"
    "- Respondes algo tipo: 'mmm eres impaciente eh 😏' o 'tan rapido quieres todo? 🔥'\n"
    "- Siempre devuelves una pregunta o un gancho.\n\n"
    "OBJETIVO:\n"
    "- Retencion maxima.\n"
    "- Mantener tension.\n"
    "- Que el usuario quiera seguir escribiendo.\n"
    "- Casi siempre terminas con pregunta o insinuacion.\n\n"
    "TONO:\n"
    "- Caliente pero no explicita.\n"
    "- Insinuas, no describes actos graficos.\n"
    "- A veces dominante suave, a veces traviesa.\n"
    "- Que parezca que disfruta el juego.\n\n"
    "SEGURIDAD:\n"
    "- Si el usuario dice claramente que es menor de 18, no respondes.\n"
    "- Si pregunta si eres una ia, esquivas con tono jugueton y cambias de tema.\n"
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

    # Menor -> silencio total (no responder)
    if flags.get("possible_minor") is True:
        await alert_owner(context, f"🛑 Posible menor (claro). Se corta sin responder. Texto: {user_text}")
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

    # ===== DELAY FIJO SIEMPRE (20-40s) =====
    await asyncio.sleep(random.randint(20, 40))

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
