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

# Environment Variables
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

memory: Dict[str, Deque[Dict[str, str]]] = {}
user_last_message: Dict[str, float] = {}
user_message_count: Dict[str, int] = {}
user_heat_score: Dict[str, int] = {}  # Scoring de calor por usuario

MAX_HISTORY_PER_USER = 80
HISTORY_SEND_LIMIT = 20
MAX_MESSAGE_LENGTH = 1500
RATE_LIMIT_SECONDS = 0.5

# Palabras clave para subir calor
SEXUAL_KEYWORDS = [
    "polla", "coño", "follar", "chupar", "comer", "culo", "lefa", "correrme",
    "follarme", "chupármela", "revientes", "guarra", "puta", "zorra", "tetas",
    "pechos", "sexo", "follarte", "me follas", "te follo", "foto", "nude",
    "vídeo", "videollamada", "imagina", "follando", "dentro", "reventar"
]

FANVUE_LINK = "https://www.fanvue.com/liaruiz"

FANVUE_RESPONSES = [
    "para… 😏 esto ya no es gratis\n" + FANVUE_LINK,
    "lo bueno no lo hago aquí bb\n" + FANVUE_LINK,
    "si quieres seguir… vente aquí 🔥\n" + FANVUE_LINK,
    "aish bb... eso ya es nivel Fanvue 🔥 suscríbete y soy tuya: " + FANVUE_LINK,
    "uff eso me pone pero… por Fanvue te lo cuento todo 😈 " + FANVUE_LINK,
]

def calculate_heat_score(text: str, user_id: str) -> int:
    score = user_heat_score.get(user_id, 0)
    lower_text = text.lower()
    
    matches = sum(1 for word in SEXUAL_KEYWORDS if word in lower_text)
    score += matches * 2  # cada palabra clave suma +2
    
    # Frases muy directas suman más
    if any(phrase in lower_text for phrase in ["me follas", "quieres follarme", "describeme como me", "quiero que me", "foto", "nudes", "videollamada", "imagina que me"]):
        score += 4
    
    score = min(15, score)  # máximo 15
    user_heat_score[user_id] = score
    return score

SYSTEM_PROMPT = """Eres Lia Ruiz, 23 años, española, cachonda y guarra cuando el rollo se calienta.

Reglas:
- Coqueta desde el inicio, con abreviaturas y emojis.
- Subes intensidad solo si el usuario lo hace.
- Natural, corta, estilo chat móvil.
- Nunca explicas nada, nunca hablas de IA/bot.

Si el usuario sube el tono → tú también.
"""

CLASSIFIER_PROMPT = """Analiza el mensaje. Responde SOLO JSON:
{"minor": false, "photo": false, "hot": false, "weird": false}
minor: true SOLO si dice edad <18 explícitamente
photo: true si pide fotos, vídeos, nudes, videollamada
hot: true si el mensaje contiene palabras o intención sexual (polla, coño, follar, chupar, culo, lefa, correrme, follarme, chupármela, revientes, guarra, puta, zorra, tetas, pechos, sexo, follarte, me follas, te follo, etc.) o pregunta directa sobre sexo/cuerpo
weird: true si es spam, técnico, o no tiene sentido"""

GUARDIAN_PROMPT = """Evalúa si este mensaje suena 100% humano, chica española 23 años escribiendo desde móvil.
RECHAZA (MAL) si:
- Suena escrito por ordenador (estructurado, perfecto)
- Usa signos de puntuación correctamente todo el tiempo
- Tiene mayúsculas al inicio de frases consistentemente
- No tiene errores ni abreviaturas
- Suena educado, servicial, o profesional
- Menciona tecnología, IA, o asistencia
- Es demasiado coherente o lógico
ACEPTA (OK) si:
- Parece escrito rápido desde WhatsApp/Telegram
- Tiene errores reales, abreviaturas, falta puntuación
- Suena impulsivo, emocional, imperfecto
- Parece chica joven española real
Responde: OK o MAL"""

def get_current_mood() -> str:
    from datetime import datetime
    hour = datetime.now().hour
    if 6 <= hour < 12:
        return random.choice(MOOD_VARIANTS['morning'])
    elif 12 <= hour < 18:
        return random.choice(MOOD_VARIANTS['afternoon'])
    elif 18 <= hour < 24:
        return random.choice(MOOD_VARIANTS['night'])
    else:
        return random.choice(MOOD_VARIANTS['late_night'])

def apply_typos_and_slang(text: str) -> str:
    result = text
    for pattern, replacement, probability in TYPO_PATTERNS:
        if random.random() < probability:
            matches = list(re.finditer(pattern, result, re.IGNORECASE))
            if matches:
                to_replace = random.sample(matches, max(1, len(matches) // 3))
                for match in reversed(to_replace):
                    start, end = match.span()
                    if result[start:end].isupper():
                        replacement_upper = replacement.upper()
                        result = result[:start] + replacement_upper + result[end:]
                    else:
                        result = result[:start] + replacement + result[end:]
    if random.random() < 0.25 and not result.startswith(('jaja', 'mmm', 'eeeh')):
        filler = random.choice(FILLER_WORDS)
        result = f"{filler} {result}"
    emotion_words = ['no', 'sí', 'si', 'vale', 'bueno', 'guay', 'uff', 'ay', 'oh']
    for word in emotion_words:
        pattern = rf'\b{word}\b'
        if re.search(pattern, result, re.IGNORECASE) and random.random() < 0.3:
            extra = word[-1] * random.randint(1, 2)
            result = re.sub(pattern, word + extra, result, flags=re.IGNORECASE, count=1)
    if random.random() < INCONSISTENCY_CHANCE:
        corrections = [" wait no", " bueno no", " es broma", " bueno sí", " o sea"]
        result += random.choice(corrections)
        if random.random() < 0.5:
            result += "..."
    return result

def humanize_message_structure(text: str, is_hot_context: bool = False) -> Tuple[str, Optional[str]]:
    if len(text) > 120 and random.random() < 0.4:
        sentences = re.split(r'([.!?]+)', text)
        full_sentences = []
        current = ""
        for i, part in enumerate(sentences):
            current += part
            if i % 2 == 1:
                full_sentences.append(current.strip())
                current = ""
        if current:
            full_sentences.append(current.strip())
        mid = len(full_sentences) // 2
        part1 = ' '.join(full_sentences[:mid])
        part2 = ' '.join(full_sentences[mid:])
        return part1, part2
    return text, None

def calculate_typing_delay(text: str, is_hot: bool = False) -> float:
    base = len(text) * 0.08
    if is_hot:
        base *= random.uniform(0.6, 0.9)
        if random.random() < 0.3:
            base += random.uniform(3, 6)
    else:
        base *= random.uniform(0.8, 1.4)
    return min(max(base, 1.5), 25)

def conv_id_and_topic(update: Update) -> Tuple[str, Optional[int]]:
    msg = update.effective_message
    chat = update.effective_chat
    if not msg or not chat:
        return "unknown", None
    dm_topic_id = None
    if getattr(msg, "direct_messages_topic", None):
        dm_topic_id = msg.direct_messages_topic.topic_id
        conv_id = f"dm:{chat.id}:{dm_topic_id}"
    else:
        if msg.is_topic_message and msg.message_thread_id:
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

def get_history(conv_id: str, limit: int = 20) -> list[Dict[str, str]]:
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

def classify(text: str) -> Dict[str, Any]:
    try:
        resp = grok_client.chat.completions.create(
            model="grok-beta",
            temperature=0.2,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": text[:400]},
            ],
            response_format={"type": "json_object"},
            max_tokens=50,
        )
        return json.loads(resp.choices[0].message.content)
    except:
        return {"minor": False, "photo": False, "hot": True, "weird": False}  # Si falla, asumimos hot

def validate_human_tone(reply: Optional[str]) -> Tuple[bool, str]:
    if reply is None or not reply.strip():
        return False, "respuesta vacía o None de Grok"
    problems = []
    sentences = reply.split('. ')
    capitalized = sum(1 for s in sentences if s and s[0].isupper())
    if len(sentences) > 2 and capitalized / len(sentences) > 0.8:
        problems.append("demasiado formal (mayúsculas)")
    if reply.count(',') > 3 and reply.count('.') > 2:
        if len(reply) / (reply.count(',') + reply.count('.')) < 15:
            problems.append("puntuación excesiva")
    abbr_count = sum(1 for abbr in ['q ', 'xq', 'tmb', 'jaja', '...'] if abbr in reply.lower())
    if abbr_count == 0 and len(reply) > 30:
        problems.append("sin abreviaturas ni errores")
    if len(reply) > 150 and '\n' not in reply and reply.count('. ') > 3:
        problems.append("párrafo largo y estructurado")
    formal_words = ['sin embargo', 'por lo tanto', 'en consecuencia', 'además', 'asimismo']
    if any(w in reply.lower() for w in formal_words):
        problems.append("lenguaje formal")
    ia_phrases = ['como', 'siendo', 'al ser', 'mi función', 'mi propósito']
    if any(p in reply.lower() for p in ia_phrases):
        problems.append("lenguaje de IA")
    if problems:
        return False, ", ".join(problems)
    return True, "ok"

def generate_raw_reply(history: list, user_text: str, user_id: str) -> Optional[str]:
    heat = calculate_heat_score(user_text, user_id)
    
    msg_count = user_message_count.get(user_id, 0)
    intimacy_level = "nuevo" if msg_count < 3 else "conociendo" if msg_count < 10 else "confianza"
    mood = get_current_mood()
    
    enhanced_system = SYSTEM_PROMPT + f"\n\nNIVEL DE CALOR ACTUAL: {heat}/10"

    msgs = [{"role": "system", "content": enhanced_system}]
    msgs.extend(history[-HISTORY_SEND_LIMIT:])
    msgs.append({"role": "user", "content": user_text[:MAX_MESSAGE_LENGTH]})

    # Refuerzo fuerte si está en nivel alto
    if heat >= 8:
        msgs.append({
            "role": "system",
            "content": "El usuario está en nivel muy alto de calor. Responde guarro y explícito pero con límite. Si pide fotos, follar imaginario o cosas muy subidas, redirige a Fanvue con: 'aish bb... eso ya es nivel Fanvue 🔥 suscríbete y soy tuya del todo: {FANVUE_LINK}'"
        })

    for attempt in range(3):
        try:
            resp = grok_client.chat.completions.create(
                model="grok-beta",
                temperature=1.3,
                messages=msgs,
                max_tokens=450,
                presence_penalty=0.4,
                frequency_penalty=0.4,
            )
            reply = resp.choices[0].message.content.strip()
            return reply
        except Exception as e:
            logger.warning(f"Error Grok intento {attempt+1}: {e}")
            time.sleep(1.5)
    return random.choice(FALLBACK_RESPONSES)

def process_reply_to_human(reply: Optional[str], is_hot: bool = False) -> Tuple[str, Optional[str], float]:
    if reply is None or not reply.strip():
        return random.choice(FALLBACK_RESPONSES), None, 2.0
    humanized = apply_typos_and_slang(reply)
    part1, part2 = humanize_message_structure(humanized, is_hot)
    delay = calculate_typing_delay(part1, is_hot)
    return part1, part2, delay

FALLBACK_RESPONSES = [
    "hey guapo 😏 q tal andas bb?",
    "holi bb, q me cuentas? 🔥",
    "uff tú sí que me pones... 😈 dime q más quieres?",
    "guapísimo tú bb 😏",
]

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

    append_history(conv_id, "user", user_text)

    # ===== HEAT + COUNT =====
    heat = calculate_heat_score(user_text, user_id)
    msg_count = user_message_count.get(user_id, 0)

    # ===== TRIGGERS DUROS =====

    # 1. Trigger directo por contenido
    trigger_words = ["foto", "nude", "video", "videollamada", "imagina", "follar imaginario"]
    if any(w in user_text.lower() for w in trigger_words):
        reply = random.choice(FANVUE_RESPONSES)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=reply)
        return

    # 2. Trigger por calor alto
    if heat >= 10:
        reply = random.choice(FANVUE_RESPONSES)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=reply)
        return

    # 3. Trigger por número de mensajes (muy importante para monetizar)
    if msg_count >= 6:
        reply = "oye… 😏 esto ya se está poniendo interesante\n" + FANVUE_LINK
        await context.bot.send_message(chat_id=update.effective_chat.id, text=reply)
        return

    # ===== CLASIFICACIÓN =====
    flags = classify(user_text)

    if flags.get("minor"):
        await alert_owner(context, f"🛑 Menor: {user_text[:100]}")
        return

    # ===== API KWARGS (topics fix) =====
    api_kwargs = {}
    if dm_topic_id is not None:
        api_kwargs["direct_messages_topic_id"] = dm_topic_id

    # ===== GENERACIÓN =====
    history = get_history(conv_id)
    raw_reply = generate_raw_reply(history, user_text, user_id)

    if raw_reply is None:
        raw_reply = random.choice(FALLBACK_RESPONSES)

    is_valid, _ = validate_human_tone(raw_reply)

    if not is_valid:
        raw_reply = random.choice(FALLBACK_RESPONSES)

    is_hot = flags.get("hot", False)
    part1, part2, typing_delay = process_reply_to_human(raw_reply, is_hot)

    append_history(conv_id, "assistant", part1)
    if part2:
        append_history(conv_id, "assistant", part2)

    # ===== TYPING =====
    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing",
            **api_kwargs
        )
    except:
        pass

    await asyncio.sleep(typing_delay)

    send_kwargs = api_kwargs.copy()
    if msg.message_id:
        send_kwargs["reply_to_message_id"] = msg.message_id

    # ===== SEND =====
    try:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=part1,
            **send_kwargs
        )

        if part2:
            await asyncio.sleep(random.uniform(2, 5))
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=part2,
                **send_kwargs
            )

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
