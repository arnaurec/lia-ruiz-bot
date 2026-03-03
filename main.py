import os
import json
import logging
from collections import deque
from typing import Dict, Deque, Any, Tuple

from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lia-bot")

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PUBLIC_URL = os.getenv("PUBLIC_URL")
PORT = int(os.getenv("PORT", "8080"))

if not BOT_TOKEN or not OPENAI_API_KEY or not PUBLIC_URL:
    raise RuntimeError("Missing required environment variables")

client = OpenAI(api_key=OPENAI_API_KEY)

memory: Dict[str, Deque[Dict[str, str]]] = {}

def conv_id_from_update(update: Update):
    msg = update.effective_message
    chat = update.effective_chat
    dm_topic_id = None

    if msg and getattr(msg, "direct_messages_topic", None):
        dm_topic_id = msg.direct_messages_topic.topic_id
        conv_id = f"dm:{chat.id}:{dm_topic_id}"
    else:
        conv_id = f"chat:{chat.id}"

    return conv_id, dm_topic_id

def append_history(conv_id: str, role: str, content: str):
    if conv_id not in memory:
        memory[conv_id] = deque(maxlen=40)
    memory[conv_id].append({"role": role, "content": content})

def get_history(conv_id: str):
    return list(memory.get(conv_id, []))

SYSTEM_PROMPT = """
Eres Lia Ruiz. Hablas en español.
Eres coqueta, juguetona y sugerente.
Respuestas cortas.
Siempre acabas con una pregunta o algo que invite a seguir escribiendo.
No escribas textos largos.
Si detectas claramente que el usuario es menor de edad, responde:
"No puedo seguir esta conversación."
"""

CLASSIFIER_PROMPT = """
Devuelve SOLO JSON válido con estas claves:
{
"possible_minor": boolean,
"asks_photo": boolean,
"high_value": boolean,
"complicated": boolean
}
possible_minor SOLO si el usuario dice claramente que es menor de 18.
"""

def classify(text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": text}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(resp.choices[0].message.content)

def generate_reply(history, user_text):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.9,
        messages=messages
    )
    return resp.choices[0].message.content.strip()

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_message or not update.effective_message.text:
        return

    user_text = update.effective_message.text.strip()
    conv_id, dm_topic_id = conv_id_from_update(update)

    append_history(conv_id, "user", user_text)

    flags = classify(user_text)

    if flags.get("possible_minor"):
        reply = "No puedo seguir esta conversación."
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=reply,
            direct_messages_topic_id=dm_topic_id
        )
        return

    history = get_history(conv_id)
    reply = generate_reply(history, user_text)

    append_history(conv_id, "assistant", reply)

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=reply,
        direct_messages_topic_id=dm_topic_id
    )

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    webhook_path = "/webhook"
    webhook_url = f"{PUBLIC_URL}{webhook_path}"

    app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path="webhook",
        webhook_url=webhook_url
    )

if __name__ == "__main__":
    main()
