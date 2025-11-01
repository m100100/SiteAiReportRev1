import os
import logging
import asyncio
import json
import re
from collections import defaultdict
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters
import azure.cognitiveservices.speech as speechsdk
import google.generativeai as genai

# === Logging Setup ===
logger = logging.getLogger("telegram_bot")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("/home/m100100/custom_bot.log")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# === Flask Setup ===
app = Flask(__name__)

# === Environment Variables ===
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# === Gemini Setup ===
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-2.5-pro")

# === Project Tracking ===
PROJECT_LIST = ["القاهرة", "الاسكندرية"]
project_totals = defaultdict(int)

# === Gemini Subroutine ===
def extract_project_and_labor(transcript: str) -> dict:
    prompt = f"""
لديك قائمة مشاريع محددة. استخرج اسم المشروع الصحيح من النص التالي، بشرط أن يكون مطابقًا لأحد المشاريع في القائمة فقط. إذا لم يكن هناك تطابق واضح، قل "غير معروف".

قائمة المشاريع:
{PROJECT_LIST}

النص: "{transcript}"

أجب فقط بصيغة JSON بدون أي شرح أو تعليق. لا تكتب أي شيء خارج القوسين.
{{
  "project_name": "...",
  "labor_count": "..."
}}
"""

    response = gemini_model.generate_content(prompt)
    raw = response.text.strip()

    # 🧹 Remove Markdown code block if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    cleaned = match.group(1) if match else raw

    logger.debug(f"Gemini cleaned response: {cleaned}")

    try:
        data = json.loads(cleaned)
        project = data.get("project_name", "").strip()
        labor = data.get("labor_count", "").strip()

        if project == "غير معروف" or not labor or labor == "0":
            return None
        return data
    except Exception as e:
        logger.error("Gemini parsing error", exc_info=True)
        return None

# === Unified Message Handler ===
async def handle_any(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Raw update: {update.to_dict()}")

    try:
        msg = update.message
        if not msg:
            return

        transcript = None

        # === Voice Message: Transcribe First ===
        if msg.voice:
            file = await context.bot.get_file(msg.voice.file_id)
            ogg_path = f"/tmp/{msg.voice.file_id}.ogg"
            await file.download_to_drive(custom_path=ogg_path)

            wav_path = ogg_path.replace(".ogg", ".wav")
            os.system(f"ffmpeg -y -i {ogg_path} -ar 16000 -ac 1 -f wav {wav_path}")

            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
            speech_config.speech_recognition_language = "ar-EG"
            audio_input = speechsdk.AudioConfig(filename=wav_path)
            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

            phrase_list = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
            for phrase in [
                "اسم المشروع", "المشروع هو", "عنوان المشروع", "موقع المشروع",
                "عدد العمال", "العمال", "العدد", "كم عامل", "الناس اللي شغالة", "عدد الموظفين",
                "القاهرة", "الاسكندرية"
            ]:
                phrase_list.addPhrase(phrase)

            result = recognizer.recognize_once()
            if result.reason != speechsdk.ResultReason.RecognizedSpeech:
                await msg.reply_text("❌ failed in voice recognition")
                return

            transcript = result.text

        # === Text Message: Use Directly ===
        elif msg.text:
            transcript = msg.text

        # === Other Message Types: Ignore ===
        else:
            return

        # === Acknowledge Receipt ===
        await msg.reply_text("✅ Message received")

        # === Gemini Extraction ===
        data = extract_project_and_labor(transcript)
        if not data:
            await msg.reply_text("❌ failed to read correct project name and correct labor number")
            return

        # === Update Totals ===
        project = data["project_name"]
        labor = int(data["labor_count"].replace(" ", "").replace("،", "").replace("٫", "").replace("٬", ""))
        project_totals[project] += labor

        # === Send Structured JSON ===
        formatted = json.dumps(data, ensure_ascii=False, indent=2)
        await msg.reply_text(f"📊 project update:\n{formatted}")

    except Exception as e:
        await msg.reply_text("❌ Processing error. please resend your message")
        logger.error("Error in handle_any", exc_info=True)

# === Register Handler ===
telegram_app = Application.builder().token(BOT_TOKEN).build()
telegram_app.add_handler(MessageHandler(filters.ALL, handle_any))

# === Initialize the bot once ===
loop = asyncio.get_event_loop()
loop.run_until_complete(telegram_app.initialize())

# === Flask Routes ===
@app.route('/')
def home():
    logger.debug("Home route hit")
    return "SiteAiReportBot is live!"

@app.route('/webhook', methods=['POST'])
def webhook():
    logger.debug("Webhook triggered")
    try:
        update = Update.de_json(request.get_json(force=True), telegram_app.bot)
        loop.run_until_complete(telegram_app.process_update(update))
        logger.debug("Update processed")
    except Exception as e:
        logger.error("Webhook error", exc_info=True)
    return "OK", 200

@app.route('/dashboard')
def dashboard():
    rows = ""
    total = 0
    for project in PROJECT_LIST:
        count = project_totals.get(project, 0)
        rows += f"<tr><td>{project}</td><td>{count}</td></tr>"
        total += count

    html = f"""
    <html>
    <head>
        <title>Labor Dashboard</title>
        <style>
            body {{ font-family: Arial; padding: 20px; }}
            table {{ border-collapse: collapse; width: 50%; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h2>📊 Labor Dashboard</h2>
        <table>
            <tr><th>Project Name</th><th>Total Labor</th></tr>
            {rows}
            <tr><th>Total</th><th>{total}</th></tr>
        </table>
    </body>
    </html>
    """
    return html
