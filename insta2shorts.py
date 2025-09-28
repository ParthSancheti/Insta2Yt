import os
import pathlib
import logging
import asyncio
import base64
import random
import pickle
import json
import re
from datetime import datetime, timedelta, time, timezone
from typing import List, Dict, Any, Optional

import requests
from PIL import Image  # noqa: F401 (import kept for completeness; PIL not strictly used directly)
import yt_dlp
from telegram import Update, Message, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)

# Google API
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import cv2  # pip install opencv-python

# ---------------- CONFIG (keys in code as requested) ----------------
TELEGRAM_BOT_TOKEN = ""
GEMINI_API_KEY = ""
PROXY = ""  # e.g. "http://user:pass@host:port" or leave ""

DOWNLOAD_DIR = pathlib.Path("downloads")
DATA_DIR = pathlib.Path("data")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# persistence files
UPLOAD_LOG = DATA_DIR / "uploads.json"             # list of {user_id, video_id, title, category, uploaded_at_iso}
USER_LIMITS = DATA_DIR / "user_limits.json"        # {user_id: int}
USER_QUEUES = DATA_DIR / "user_queues.json"        # {user_id: [{url, scheduled_for_iso}]}
PENDING_UPLOADS = DATA_DIR / "pending_uploads.json" # {msg_id: {user_id, path, meta, category, expires_at_iso}}

DEFAULT_DAILY_LIMIT = 10

YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]

# Timezone
IST = timezone(timedelta(hours=5, minutes=30))  # Asia/Kolkata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("insta2yt")

# URL validation
YOUTUBE_RE = re.compile(r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/", re.I)
INSTA_RE = re.compile(r"(https?://)?(www\.)?instagram\.com/", re.I)

# ---------------- Small JSON persistence helpers ----------------
def _read_json(path: pathlib.Path, default):
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Failed reading %s: %s", path, e)
    return default

def _write_json(path: pathlib.Path, obj):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def add_upload_log(user_id: int, video_id: str, title: str, category: str):
    logs = _read_json(UPLOAD_LOG, [])
    logs.append({
        "user_id": user_id,
        "video_id": video_id,
        "title": title,
        "category": category,
        "uploaded_at_iso": datetime.now(IST).isoformat()
    })
    _write_json(UPLOAD_LOG, logs)

def get_week_range_ist(dt: datetime):
    dt = dt.astimezone(IST)
    start = (dt - timedelta(days=dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=7)
    return start, end

def get_user_limit(user_id: int) -> int:
    limits = _read_json(USER_LIMITS, {})
    return int(limits.get(str(user_id), DEFAULT_DAILY_LIMIT))

def set_user_limit(user_id: int, limit: int):
    limits = _read_json(USER_LIMITS, {})
    limits[str(user_id)] = int(limit)
    _write_json(USER_LIMITS, limits)

def enqueue_for_tomorrow_noon(user_id: int, url: str):
    queues = _read_json(USER_QUEUES, {})
    user_q = queues.get(str(user_id), [])
    now = datetime.now(IST)
    scheduled_dt = datetime.combine((now + timedelta(days=1)).date(), time(12, 0), tzinfo=IST)
    user_q.append({"url": url, "scheduled_for_iso": scheduled_dt.isoformat()})
    queues[str(user_id)] = user_q
    _write_json(USER_QUEUES, queues)

def dequeue_due_items() -> List[Dict[str, Any]]:
    queues = _read_json(USER_QUEUES, {})
    now = datetime.now(IST)
    due = []
    for uid, items in list(queues.items()):
        keep = []
        for it in items:
            try:
                when = datetime.fromisoformat(it["scheduled_for_iso"])
            except Exception:
                keep.append(it)
                continue
            if when <= now:
                due.append({"user_id": int(uid), "url": it["url"]})
            else:
                keep.append(it)
        queues[uid] = keep
    _write_json(USER_QUEUES, queues)
    return due

# ---------------- Daily counters (in-memory) ----------------
daily_counters: Dict[int, Dict[str, Any]] = {}  # {user_id: {"count": int, "date": date}}

def can_process_now(user_id: int) -> bool:
    today = datetime.now(IST).date()
    entry = daily_counters.get(user_id)
    if not entry or entry["date"] != today:
        daily_counters[user_id] = {"count": 0, "date": today}
    limit = get_user_limit(user_id)
    return daily_counters[user_id]["count"] < limit

def mark_processed(user_id: int):
    today = datetime.now(IST).date()
    entry = daily_counters.get(user_id)
    if not entry or entry["date"] != today:
        daily_counters[user_id] = {"count": 0, "date": today}
    daily_counters[user_id]["count"] += 1

# ---------------- yt-dlp download ----------------
def download_video(url: str, progress_callback=None) -> pathlib.Path:
    """
    Download IG or YT with yt-dlp. Merge to mp4 when needed.
    Optional: cookies + proxy
    """
    logger.info("Downloading: %s", url)
    ydl_opts = {
        "format": "bestvideo*[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(DOWNLOAD_DIR / "%(id)s.%(ext)s"),
        "progress_hooks": [progress_callback] if progress_callback else [],
        "nocheckcertificate": True,
        "noprogress": True,
    }
    if "instagram.com" in url:
        cookie_file = pathlib.Path("instagram_cookies.txt")
        if cookie_file.exists():
            ydl_opts["cookiefile"] = str(cookie_file)
        else:
            logger.warning("Instagram cookies file not found; IG downloads may fail.")
    if ("youtube.com" in url) or ("youtu.be" in url):
        y_cookie = pathlib.Path("youtube_cookies.txt")
        if y_cookie.exists():
            ydl_opts["cookiefile"] = str(y_cookie)
    if PROXY:
        ydl_opts["proxy"] = PROXY

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = pathlib.Path(ydl.prepare_filename(info))
    return filepath

# ---------------- Frame extraction (5 random frames) ----------------
def extract_frames(video_path: pathlib.Path, num_frames: int = 5) -> List[bytes]:
    frames: List[bytes] = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError("Could not open video file.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            taken = 0
            while taken < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                ok, buf = cv2.imencode(".png", frame)
                if ok:
                    frames.append(buf.tobytes())
                    taken += 1
            return frames

        duration = frame_count / fps
        positions = sorted(set([random.uniform(0, max(0.2, duration - 0.2)) for _ in range(num_frames)]))
        for t in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ret, frame = cap.read()
            if not ret:
                continue
            ok, buf = cv2.imencode(".png", frame)
            if ok:
                frames.append(buf.tobytes())
    except Exception as e:
        logger.error("extract_frames failed: %s", e)
    finally:
        cap.release()
    return frames

# ---------------- Gemini metadata ----------------
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_PROMPT = """
You are a senior YouTube Shorts metadata expert.
You will receive 5 frames from a short video. Produce a strict JSON object with keys:
- category: one of ["Romantic","Cars","Nature","Random"]
- title: 60-80 characters, catchy, keyword-rich, no emojis
- description: 150-200 words, SEO-friendly
- tags: 10-15 plain words (NO # symbols), suitable for YouTube snippet.tags
- hashtags: 8-15 hashtags starting with # for use in the description

Rules:
- category must be exactly one of the four options
- tags must be plain words (no #), lowercase preferred
- hashtags may include #shorts and general discovery
Return ONLY the JSON. No extra text.
"""

def encode_images_to_b64_list(images_bytes: List[bytes]) -> List[str]:
    return [base64.b64encode(b).decode("utf-8") for b in images_bytes]

def call_gemini(images_bytes: List[bytes]) -> Optional[Dict[str, Any]]:
    if not images_bytes or not GEMINI_API_KEY:
        return None
    parts = [{"text": GEMINI_PROMPT}]
    for b in encode_images_to_b64_list(images_bytes):
        parts.append({"inline_data": {"mime_type": "image/png", "data": b}})
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 900}
    }
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}", headers=headers, json=payload, timeout=40)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        try:
            obj = json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            obj = json.loads(m.group(0)) if m else None
        return obj
    except Exception as e:
        logger.warning("Gemini call failed: %s", e)
        return None

def fallback_metadata(title_hint: str) -> Dict[str, Any]:
    cats = ["Romantic","Cars","Nature","Random"]
    category = random.choice(cats)
    title = f"{title_hint[:70]} | {category} Short"
    description = f"{title}\nAuto generated."
    tags = [category.lower(), "shorts", "viral", "explore", "trending", "clip", "video", "best", "amazing", "wow"]
    hashtags = [f"#{t}" for t in ["shorts","viral","trending","explore","reels","fyp"]]
    return {"category": category, "title": title, "description": description, "tags": tags, "hashtags": hashtags}

def normalize_meta(obj: Optional[Dict[str, Any]], hint: str) -> Dict[str, Any]:
    if not obj:
        return fallback_metadata(hint)
    category = str(obj.get("category") or "Random").strip().title()
    if category not in {"Romantic","Cars","Nature","Random"}:
        category = "Random"
    title = str(obj.get("title") or hint or "Untitled Short")[:100]
    description = str(obj.get("description") or "")
    tags = obj.get("tags") or []
    hashtags = obj.get("hashtags") or []
    tags = [re.sub(r"^#","", str(t)).strip().lower() for t in tags if str(t).strip()]
    hashtags = [h if str(h).startswith("#") else f"#{str(h).strip()}" for h in hashtags if str(h).strip()]
    return {"category": category, "title": title, "description": description, "tags": tags, "hashtags": hashtags}

# ---------------- YouTube helpers ----------------
def load_client_secrets(category_key: str):
    files = {
        "romantic": "romantic_client_secrets.json",
        "cars": "cars_client_secrets.json",
        "nature": "nature_client_secrets.json",
        "random": "random_client_secrets.json",
    }
    fname = files.get(category_key.lower())
    if not fname or not os.path.exists(fname):
        raise ValueError(f"Client secrets file for category {category_key} not found.")
    with open(fname, "r", encoding="utf-8") as f:
        return json.load(f)

def get_youtube_service(client_secrets, category_key: str):
    token_file = f"token_{category_key.lower()}.pickle"
    creds = None
    if os.path.exists(token_file):
        with open(token_file, "rb") as f:
            creds = pickle.load(f)
    if not creds or not getattr(creds, "valid", False):
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.warning("Token refresh failed for %s: %s", category_key, e)
                creds = None
        if not creds:
            flow = InstalledAppFlow.from_client_config(client_secrets, YOUTUBE_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_file, "wb") as f:
            pickle.dump(creds, f)
        try:
            os.chmod(token_file, 0o600)
        except Exception:
            pass
    return build("youtube", "v3", credentials=creds)

def upload_youtube(video_path: pathlib.Path, meta: Dict[str, Any], category_key: str,
                   msg: Optional[Message], context: Optional[ContextTypes.DEFAULT_TYPE],
                   loop: asyncio.AbstractEventLoop) -> str:
    client_secrets = load_client_secrets(category_key)
    youtube = get_youtube_service(client_secrets, category_key)
    clean_tags = [t.lstrip("#") for t in meta.get("tags", [])]
    desc = meta["description"].strip()
    if meta.get("hashtags"):
        desc = f"{desc}\n\n{' '.join(meta['hashtags'])}"
    body = {
        "snippet": {"title": meta["title"], "description": desc, "tags": clean_tags, "categoryId": "22"},
        "status": {"privacyStatus": "public"}
    }
    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    try:
        while response is None:
            status, response = request.next_chunk()
            if status and msg and context:
                p = status.progress() * 100
                bars = int(p / 10)
                bar = '█' * bars + '▒' * (10 - bars)
                coro = msg.edit_text(f"🚀 Uploading... [{bar}] {p:.2f}%")
                asyncio.run_coroutine_threadsafe(coro, loop)
        return response.get("id")
    except HttpError as e:
        try:
            error_details = json.loads(e.content.decode())
        except Exception:
            error_details = {}
        if error_details.get('error', {}).get('errors', [{}])[0].get('reason') == 'youtubeSignupRequired':
            raise ValueError("Create a YouTube channel first: https://www.youtube.com/create_channel")
        raise

def get_uploads_playlist_id(youtube):
    channels = youtube.channels().list(part="contentDetails", mine=True).execute()
    items = channels.get("items", [])
    if not items:
        return None
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

def list_my_videos(youtube, max_results=50):
    uploads_id = get_uploads_playlist_id(youtube)
    if not uploads_id:
        return []
    videos = []
    token = None
    while True:
        pl = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_id,
            maxResults=min(50, max_results),
            pageToken=token
        ).execute()
        for item in pl.get("items", []):
            videos.append({
                "title": item["snippet"]["title"],
                "videoId": item["contentDetails"]["videoId"]
            })
            if len(videos) >= max_results:
                return videos
        token = pl.get("nextPageToken")
        if not token:
            return videos

def get_video_stats(youtube, video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    stats = {}
    if not video_ids:
        return stats
    for i in range(0, len(video_ids), 50):
        resp = youtube.videos().list(part="statistics", id=",".join(video_ids[i:i+50])).execute()
        for it in resp.get("items", []):
            stats[it["id"]] = it.get("statistics", {})
    return stats

# ---------------- Allowed commands/URLs gate ----------------
def is_command_or_url(text: str) -> bool:
    text = (text or "").strip()
    if text.startswith("/"):
        return text.split()[0] in {"/report","/ping","/stats","/setlimt"}
    if YOUTUBE_RE.search(text) or INSTA_RE.search(text):
        return True
    return False

# ---------------- Pending upload (category confirm window) ----------------
def save_pending(msg_id: int, payload: Dict[str, Any]):
    data = _read_json(PENDING_UPLOADS, {})
    data[str(msg_id)] = payload
    _write_json(PENDING_UPLOADS, data)

def pop_pending(msg_id: int) -> Optional[Dict[str, Any]]:
    data = _read_json(PENDING_UPLOADS, {})
    obj = data.pop(str(msg_id), None)
    _write_json(PENDING_UPLOADS, data)
    return obj

def get_pending(msg_id: int) -> Optional[Dict[str, Any]]:
    data = _read_json(PENDING_UPLOADS, {})
    return data.get(str(msg_id))

# ---------------- Handlers ----------------
async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Bot is live.")

async def setlimt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    parts = (update.message.text or "").strip().split()
    if len(parts) != 2 or not parts[1].isdigit():
        await update.message.reply_text("Usage: /setlimt <number_per_day>")
        return
    n = max(1, min(100, int(parts[1])))
    set_user_limit(user_id, n)
    await update.message.reply_text(f"✅ Daily limit set to {n}.")

async def report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logs = _read_json(UPLOAD_LOG, [])
    now = datetime.now(IST)
    start, end = get_week_range_ist(now)
    this_week = []
    for row in logs:
        try:
            t = datetime.fromisoformat(row["uploaded_at_iso"])
        except Exception:
            continue
        if start <= t.astimezone(IST) < end:
            this_week.append(row)
    if not this_week:
        await update.message.reply_text("No uploads this week yet.")
        return
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for r in this_week:
        by_cat.setdefault(r["category"], []).append(r)
    lines = [f"📊 This week ({start.date()} to {(end - timedelta(days=1)).date()}):", ""]
    for cat, items in by_cat.items():
        lines.append(f"• {cat}:")
        for it in items:
            lines.append(f"  - {it['title']} (https://youtu.be/{it['video_id']})")
        lines.append("")
    await update.message.reply_text("\n".join(lines))

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logs = _read_json(UPLOAD_LOG, [])
    if not logs:
        await update.message.reply_text("No uploads yet.")
        return
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for r in logs[-200:]:
        by_cat.setdefault(r["category"].lower(), []).append(r)
    lines = ["📈 Stats (recent uploads):", ""]
    for cat_key, items in by_cat.items():
        try:
            yt = get_youtube_service(load_client_secrets(cat_key), cat_key)
        except Exception as e:
            lines.append(f"{cat_key.title()}: (auth error: {e})")
            continue
        ids = [r["video_id"] for r in items]
        s = get_video_stats(yt, ids)
        lines.append(f"{cat_key.title()}:")
        for r in items:
            views = s.get(r["video_id"], {}).get("viewCount", "0")
            lines.append(f"- {r['title']} | {views} views | {r['category']}")
        lines.append("")
    await update.message.reply_text("\n".join(lines))

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    # Ignore anything that isn't one of the approved commands or an IG/YT link
    if not is_command_or_url(text):
        return

    if text.startswith("/"):
        if text.startswith("/report"):
            return await report(update, context)
        if text.startswith("/ping"):
            return await ping(update, context)
        if text.startswith("/stats"):
            return await stats(update, context)
        if text.startswith("/setlimt"):
            return await setlimt(update, context)
        return

    # It's a URL
    user_id = update.effective_chat.id
    url = text

    if not (YOUTUBE_RE.search(url) or INSTA_RE.search(url)):
        return

    if not can_process_now(user_id):
        enqueue_for_tomorrow_noon(user_id, url)
        limit = get_user_limit(user_id)
        await update.message.reply_text(f"⏸️ Daily limit {limit} reached. Queued for tomorrow 12:00 PM IST.")
        return

    await process_url(update, context, url)

async def process_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str):
    user_id = update.effective_chat.id
    loop = asyncio.get_running_loop()
    msg = await update.message.reply_text("⏳ Downloading...")

    try:
        downloaded = download_video(url)
    except Exception as e:
        await msg.edit_text(
            "❌ Download failed: " + str(e) + "\n"
            "Tip: set PROXY in code if you're behind a proxy, or add youtube_cookies.txt / instagram_cookies.txt."
        )
        return

    await msg.edit_text("🔍 Generating metadata...")
    try:
        images_bytes = extract_frames(downloaded, num_frames=5)
        meta_raw = call_gemini(images_bytes)
        meta = normalize_meta(meta_raw, downloaded.stem)
    except Exception as e:
        logger.warning("Metadata generation failed: %s", e)
        meta = fallback_metadata(downloaded.stem)

    # Category correction buttons; auto proceed after 60s
    categories = ["Romantic","Cars","Nature","Random"]
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(cat + (" ✅" if cat == meta["category"] else ""), callback_data=f"cat::{cat}")]
        for cat in categories
    ])
    prompt = (
        f"Proposed category: *{meta['category']}*\n"
        f"Title: {meta['title']}\n\n"
        "Change category below if needed. Upload starts automatically in 60 seconds."
    )
    sent = await update.message.reply_text(prompt, reply_markup=keyboard, parse_mode="Markdown")

    pending = {
        "user_id": user_id,
        "path": str(downloaded),
        "meta": meta,
        "category": meta["category"],
        "expires_at_iso": (datetime.now(IST) + timedelta(seconds=60)).isoformat(),
        "source_msg_id": sent.message_id
    }
    data = _read_json(PENDING_UPLOADS, {})
    data[str(sent.message_id)] = pending
    _write_json(PENDING_UPLOADS, data)

    asyncio.create_task(auto_upload_after_timeout(context, sent.chat_id, sent.message_id))

async def auto_upload_after_timeout(context: ContextTypes.DEFAULT_TYPE, chat_id: int, msg_id: int):
    await asyncio.sleep(60)
    pending = get_pending(msg_id)
    if not pending:
        return
    await proceed_upload(context, chat_id, msg_id)

async def proceed_upload(context: ContextTypes.DEFAULT_TYPE, chat_id: int, msg_id: int):
    loop = asyncio.get_running_loop()
    obj = pop_pending(msg_id)
    if not obj:
        return
    user_id = obj["user_id"]
    video_path = pathlib.Path(obj["path"])
    meta = obj["meta"]
    category_key = (obj.get("category") or meta.get("category") or "Random").lower()
    if category_key not in {"romantic","cars","nature","random"}:
        category_key = "random"

    mark_processed(user_id)

    try:
        msg = await context.bot.send_message(chat_id, "🚀 Starting upload...")
    except Exception:
        return

    try:
        vid = upload_youtube(video_path, meta, category_key, msg, context, loop)
        await context.bot.send_message(
            chat_id,
            f"🎉 Uploaded!\n📌 Title: {meta['title']}\n🏷️ Category: {category_key.title()}\n🔗 https://youtu.be/{vid}"
        )
        add_upload_log(user_id, vid, meta["title"], category_key.title())
    except Exception as e:
        await context.bot.send_message(chat_id, f"❌ Upload failed: {e}")
        return
    finally:
        try:
            if video_path.exists():
                video_path.unlink()
        except Exception:
            pass

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    if not data.startswith("cat::"):
        return
    new_cat = data.split("::", 1)[1]
    pending = get_pending(query.message.message_id)
    if not pending:
        await query.edit_message_text("No pending upload.")
        return
    pending["category"] = new_cat
    save_pending(query.message.message_id, pending)
    await query.edit_message_text(
        f"Category set to *{new_cat}*.\nUpload will start automatically when the timer finishes.",
        parse_mode="Markdown"
    )

# ---------------- JobQueue: process due queued items ----------------
async def process_url_for_chat(context: ContextTypes.DEFAULT_TYPE, chat_id: int, url: str):
    class TempMessage:
        def __init__(self, bot, chat_id):
            self.chat_id = chat_id
            self._bot = bot
        async def reply_text(self, text, **kwargs):
            return await self._bot.send_message(self.chat_id, text, **kwargs)

    class TempUpdate:
        def __init__(self, bot, chat_id):
            self.effective_chat = type("C", (), {"id": chat_id})
            self.message = TempMessage(bot, chat_id)

    tmp_update = TempUpdate(context.bot, chat_id)
    await process_url(tmp_update, context, url)

async def queue_watcher_job(context: ContextTypes.DEFAULT_TYPE):
    due_items = dequeue_due_items()
    for item in due_items:
        chat_id = item["user_id"]
        url = item["url"]
        try:
            await context.bot.send_message(chat_id, f"⏳ Processing queued link: {url}")
            await process_url_for_chat(context, chat_id, url)
        except Exception as e:
            logging.exception("Queue watcher failed for %s: %s", url, e)

# ---------------- Main ----------------
def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in code.")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Only the commands you wanted
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("report", report))
    app.add_handler(CommandHandler("setlimt", setlimt))

    # Category correction button callbacks
    app.add_handler(CallbackQueryHandler(on_button))

    # Only react to texts that are allowed (URLs). Everything else is ignored.
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # background job to check queued URLs
    app.job_queue.run_repeating(queue_watcher_job, interval=60, first=10)

    logger.info("Bot starting...")
    app.run_polling()

if __name__ == "__main__":
    main()
