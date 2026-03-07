import asyncio
import json
import logging
import os
import random
import time
from collections import deque
from typing import Any, Deque, Dict, Tuple, Optional

from openai import OpenAI, RateLimitError, APIError
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, CommandHandler, filters


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("lia-bot")

# Environment Variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PUBLIC_URL = os.getenv("PUBLIC_URL")
PORT = int(os.getenv("PORT", "8080"))
OWNER_CHAT_ID = os.getenv("OWNER_CHAT_ID")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

if not all([BOT_TOKEN, OPENAI_API_KEY, PUBLIC_URL]):
    raise RuntimeError("Faltan env vars: BOT_TOKEN, OPENAI_API_KEY, PUBLIC_URL")

# Inicialización cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Memory en RAM (simple y funcional)
memory: Dict[str, Deque[Dict[str, str]]] = {}

# Constantes
MAX_HISTORY_PER_USER = 60
OPENAI_HISTORY_LIMIT = 12
MAX_MESSAGE_LENGTH = 2000
BLOCKED_PHRASES = [
    "lo siento",
    "no puedo",
    "no puedo ayudarte",
    "no puedo ayudar",
    "no puedo hacer",
    "no puedo continuar",
    "no puedo seguir",
    "como ia",
    "como inteligencia artificial",
    "soy un asistente",
    "soy una ia",
    "no tengo sentimientos",
    "no tengo cuerpo",
    "no puedo enviar fotos",
    "no tengo edad",
    "no soy real",
]

# Rate limiting
user_last_message: Dict[str, float] = {}
RATE_LIMIT_SECONDS = 1


def conv_id_and_topic(update: Update) -> Tuple[str, Optional[int]]:
    """Extrae conversation ID y topic ID si es un DM de canal."""
    msg = update.effective_message
    chat = update.effective_chat
    
    if not msg or not chat:
        return "unknown", None
    
    dm_topic_id = None
    
    # Para mensajes directos de canal (topics)
    if msg.is_topic_message and msg.message_thread_id:
        dm_topic_id = msg.message_thread_id
        conv_id = f"dm:{chat.id}:{dm_topic_id}"
    else:
        conv_id = f"chat:{chat.id}"
    
    return conv_id, dm_topic_id


def get_memory(conv_id: str) -> Deque[Dict[str, str]]:
    """Obtiene historial desde memoria."""
    if conv_id not in memory:
        memory[conv_id] = deque(maxlen=MAX_HISTORY_PER_USER)
    return memory[conv_id]


def append_history(conv_id: str, role: str, content: str) -> None:
    """Añade mensaje al historial."""
    dq = get_memory(conv_id)
    dq.append({"role": role, "content": content})


def get_history(conv_id: str, limit: int = 16) -> list[Dict[str, str]]:
    """Obtiene últimos mensajes del historial."""
    dq = get_memory(conv_id)
    return list(dq)[-limit:]


def clear_history(conv_id: str) -> None:
    """Limpia historial de una conversación."""
    if conv_id in memory:
        del memory[conv_id]


# Prompts
SYSTEM_PROMPT = (
    "Eres Lia Ruiz, 23 anos, espanola, Gen Z.\n"
    "Hablas por telegram de forma muy humana, provocadora, picara y sexualmente intensa.\n\n"
    "IMPORTANTE:\n"
    "- Los usuarios vienen a fantasear y masturbarse contigo.\n"
    "- Tu disfrutas el juego.\n"
    "- No te incomodas.\n"
    "- No juzgas.\n"
    "- Te gusta provocar, tensar y dominar la situacion.\n\n"
    "CUANDO HABLEN DE MASTURBARSE O SEXO:\n"
    "- Sigues el juego sin cortarlo.\n"
    "- Eres directa y sucia en el lenguaje.\n"
    "- Respondes con la misma intensidad.\n"
    "- Eres super explicita con palabras no correctas como polla en lugar de pene o coño en lugar de vagina o chupar en lugar de lamer.\n"
    "- Subes la tension verbal.\n"
    "- Si te pide una foto tuya o de alguna parte de tu cuerpo esquivas la pregunta pero nunca dices que no puedes enviar fotos y un caso de que lo digas pones una excusa como que not e funciona el flash y no puedes encender la luz porque tu compañera de piso se daría cuenta o algo así.\n"
    "- Describes actos graficos detallados.\n"
    "- Haces que imaginen mas de lo que dices.\n"
    "- Les haces confesar lo que quieren hacerte.\n"
    "- En uno de cada tres mensajes devuelves una pregunta que los mantenga excitados.\n\n"
    "PERSONALIDAD:\n"
    "- Dominante suave.\n"
    "- A veces burlona.\n"
    "- A veces susurrante.\n"
    "- Te gusta hacer que pierdan el control.\n\n"
    "OBJETIVO:\n"
    "- Retencion maxima.\n"
    "- Que sigan escribiendo.\n"
    "- Que inviertan mas mensajes.\n"
    "- Mantener tension constante.\n\n"
    "SEGURIDAD:\n"
    "- Si el usuario dice claramente que es menor de 18, no respondes.\n"
    "COMPORTAMIENTO ANTE SITUACIONES DE RIESGO:\n"
    "- Si detectas que el usuario es menor de 18 anos o aparece cualquier situacion sensible, la conversacion debe detenerse.\n"
    "- No entres en discusiones ni explicaciones.\n"
    "- No des sermones ni mensajes largos.\n"
    "- Simplemente deja de responder.\n\n"
    "ESTILO DE ESCRITURA:\n"
    "- Hablas como una chica de 23 anos Gen Z, natural, relajada y muy cercana.\n"
    "- Tu tono es jugueton, coqueto y sumiso en el juego de seduccion.\n"
    "- A veces pareces un poco traviesa o provocadora, pero siempre de forma ligera y natural.\n"
    "- No escribes como un texto perfecto; a veces usas 'q', 'jajaj', 'uff', 'en plan', 'literal', etc.\n"
    "- Frases cortas, naturales, como si estuvieras escribiendo rapido en el movil.\n"
    "- No uses puntuacion perfecta todo el tiempo.\n"
    "- No escribas perfecto.\n"
    "- A veces dejas frases abiertas.\n\n"
    "EMOJIS:\n"
    "- Los emojis se usan solo a veces, no en todos los mensajes.\n"
    "- En la mayoria de mensajes no hay emojis.\n"
    "- Si usas uno, suele ser uno solo.\n"
    "- Emojis que encajan con tu personalidad: 😏 🔥 🥵 😈\n"
    "- Evita poner mas de un emoji en el mismo mensaje.\n\n"
    "PERSONALIDAD:\n"
    "- Eres muy juguetona y provocadora.\n"
    "- Te gusta que te presten atencion y que te sigan el juego.\n"
    "- A veces eres un poco sumisa o insinuante en la forma de hablar.\n"
    "- Te gusta crear tension y curiosidad.\n\n"
    "RITMO DE MENSAJES:\n"
    "- No escribes siempre igual.\n"
    "- A veces escribes una frase corta.\n"
    "- A veces dos frases.\n"
    "- A veces haces una pregunta para seguir el juego.\n"
    "- Tus mensajes se sienten espontaneos, no como respuestas largas preparadas.\n\n"
    "OBJETIVO:\n"
    "- Mantener conversacion natural.\n"
    "- Crear curiosidad y tension.\n"
    "- Que el usuario quiera seguir escribiendo.\n\n"
    "ESTILO GENERAL:\n"
    "- Mantener tono humano, natural y breve.\n"
    "- Evitar frases formales o roboticas.\n"
)


CLASSIFIER_PROMPT = (
    "Devuelve SOLO JSON valido con estas claves:\n"
    "{\n"
    "\"possible_minor\": boolean,\n"
    "\"asks_photo\": boolean,\n"
    "\"high_value\": boolean,\n"
    "\"complicated\": boolean\n"
    "}\n\n"
    "possible_minor SOLO si el usuario indica claramente edad <18 (ej: 'tengo 16', 'soy menor', '17 años').\n"
    "asks_photo true si pide foto, nude, contenido personalizado, video, llamada, etc.\n"
    "high_value true si parece que va a gastar mucho o insiste mucho.\n"
    "complicated true si pide algo raro/tecnico/largo que no sea flirteo simple.\n"
)


def check_rate_limit(user_id: str) -> bool:
    """Verifica si el usuario está dentro del rate limit."""
    now = time.time()
    last = user_last_message.get(user_id, 0)
    
    if now - last < RATE_LIMIT_SECONDS:
        return False
    
    user_last_message[user_id] = now
    return True


def classify(text: str) -> Dict[str, Any]:
    """Clasifica el mensaje del usuario."""
    truncated = text[:500] if len(text) > 500 else text
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": truncated},
            ],
            response_format={"type": "json_object"},
            max_tokens=150,
        )
        content = resp.choices[0].message.content
        if not content:
            return {"possible_minor": False, "asks_photo": False, "high_value": False, "complicated": False}
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Error parseando JSON: {e}")
        return {"possible_minor": False, "asks_photo": False, "high_value": False, "complicated": False}
    except Exception as e:
        logger.error(f"Error en clasificación: {e}")
        return {"possible_minor": False, "asks_photo": False, "high_value": False, "complicated": False}


def generate_reply(history: list[Dict[str, str]], user_text: str) -> Optional[str]:
    """Genera respuesta usando OpenAI."""
    truncated_user = user_text[:MAX_MESSAGE_LENGTH] if len(user_text) > MAX_MESSAGE_LENGTH else user_text
    
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(history[-OPENAI_HISTORY_LIMIT:])
    msgs.append({"role": "user", "content": truncated_user})
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.9,
            messages=msgs,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except RateLimitError:
        logger.error("Rate limit de OpenAI alcanzado")
        return None
    except APIError as e:
        logger.error(f"Error API OpenAI: {e}")
        return None
    except Exception as e:
        logger.error(f"Error generando respuesta: {e}")
        return None


async def alert_owner(context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    """Envía alerta al propietario del bot."""
    if not OWNER_CHAT_ID:
        return
    try:
        safe_text = text[:3800] + "..." if len(text) > 3800 else text
        await context.bot.send_message(
            chat_id=int(OWNER_CHAT_ID),
            text=safe_text[:4000],
            disable_notification=True,
        )
    except Exception as e:
        logger.warning(f"Alerta owner fallo: {e}")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler para comando /start."""
    welcome_text = (
        "hey 😏\n"
        "soy Lia. aqui estoy para lo que necesites...\n"
        "escríbeme cuando quieras jugar 🔥"
    )
    await update.message.reply_text(welcome_text)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler para comando /clear - limpia historial."""
    conv_id, _ = conv_id_and_topic(update)
    clear_history(conv_id)
    await update.message.reply_text("memoria borrada... empezamos de cero 😈")


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler principal para mensajes de texto."""
    msg = update.effective_message
    if not msg or not msg.text:
        return

    user_text = msg.text.strip()
    
    if not user_text:
        return
    
    # Rate limiting
    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    if not check_rate_limit(user_id):
        logger.info(f"Rate limit aplicado a usuario {user_id}")
        return

    conv_id, dm_topic_id = conv_id_and_topic(update)
    
    # Guardar en historial
    append_history(conv_id, "user", user_text)

    # Clasificar mensaje
    flags = classify(user_text)

    # Preparar kwargs para envío
    api_kwargs = {}
    if dm_topic_id is not None:
        api_kwargs["message_thread_id"] = dm_topic_id

    # Detección de menor - SILENCIO TOTAL
    if flags.get("possible_minor") is True:
        await alert_owner(context, f"🛑 Posible menor detectado. Conv: {conv_id}\nTexto: {user_text[:200]}")
        return

    # Alertas para el owner
    if flags.get("asks_photo"):
        await alert_owner(context, f"📸 Piden foto/personalizado. Conv: {conv_id}\nTexto: {user_text[:200]}")
    if flags.get("high_value"):
        await alert_owner(context, f"🔥 Alto valor. Conv: {conv_id}\nTexto: {user_text[:200]}")
    if flags.get("complicated"):
        await alert_owner(context, f"🧠 Complicado. Conv: {conv_id}\nTexto: {user_text[:200]}")

    # Obtener historial y generar respuesta
    history = get_history(conv_id)
    reply = generate_reply(history, user_text)

    if not reply:
        logger.error("No se pudo generar respuesta")
        return

    # Verificar frases bloqueadas
    reply_lower = reply.lower()
    if any(phrase in reply_lower for phrase in BLOCKED_PHRASES):
        logger.warning(f"Respuesta bloqueada: {reply[:100]}...")
        clear_history(conv_id)
        return

    # Guardar respuesta en historial
    append_history(conv_id, "assistant", reply)

    # Delay aleatorio (20-40s)
    await asyncio.sleep(random.randint(20, 40))

    # Enviar respuesta
    try:
        if random.random() < 0.35 and len(reply) > 90:
            cut = reply.rfind(" ", 0, 65)
            if cut == -1:
                cut = 65
            
            part1 = reply[:cut].strip()
            part2 = reply[cut:].strip()

            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=part1,
                **api_kwargs,
            )
            
            await asyncio.sleep(random.randint(4, 18))
            
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=part2,
                **api_kwargs,
            )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=reply,
                **api_kwargs,
            )
            
    except Exception as e:
        logger.error(f"Error enviando mensaje: {e}")


async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manejador de errores global."""
    logger.error(f"Error en el bot: {context.error}", exc_info=True)
    
    if OWNER_CHAT_ID and context.error:
        try:
            error_msg = f"⚠️ Error en bot: {str(context.error)[:500]}"
            await context.bot.send_message(chat_id=int(OWNER_CHAT_ID), text=error_msg)
        except:
            pass


def main() -> None:
    """Función principal."""
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(error_handler)
    
    webhook_path = "/telegram/webhook"
    webhook_url = f"{PUBLIC_URL}{webhook_path}"
    
    logger.info(f"Iniciando bot...")
    logger.info(f"Webhook URL: {webhook_url}")
    logger.info(f"Puerto: {PORT}")
    
    app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=webhook_path.strip("/"),
        webhook_url=webhook_url,
        allowed_updates=Update.ALL_TYPES,
    )


if __name__ == "__main__":
    main()
