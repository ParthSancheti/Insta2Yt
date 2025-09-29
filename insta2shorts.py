# -*- coding: utf-8 -*-
import os
import pathlib
import logging
import asyncio
import base64
import random
import pickle
import json
import re
from contextlib import contextmanager
from datetime import datetime, timedelta, time, timezone
from typing import List, Dict, Any, Optional

import requests
from PIL import Image  # noqa: F401
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

# ---------------- CONFIG (original kept; added OWNER_CHAT_ID, GITHUB_URL) ----------------
TELEGRAM_BOT_TOKEN = "8421836345:AAGJxrPx4PwOJRCQYZ5djLuoJOhoIw651as"
GEMINI_API_KEY = "AIzaSyCFFb178udfeXHaVCU2BKiKsDOqlb2Tka4"

# Optional HTTP proxy (leave "" for none). Example: "http://user:pass@host:port"
PROXY = ""

# Owner chat to receive 4-hour auto finds (set your Telegram numeric user id)
OWNER_CHAT_ID = -1002984074793

  # e.g. 123456789

# Optional GitHub URL to show in /start
GITHUB_URL = ""  # e.g. "https://github.com/you/repo"

# Directories (YOUR layout)
TEMP_DIR = pathlib.Path("Temp")
PICKLES_DIR = pathlib.Path("Pickles")    # tokens live here
JSON_DIR = pathlib.Path("Json")           # client secrets + state live here
for p in (TEMP_DIR, PICKLES_DIR, JSON_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Optional single fallback client secret if a category JSON is missing
DEFAULT_CLIENT_SECRETS = JSON_DIR / "main_client_secrets.json"

# Persistence files
UPLOAD_LOG = JSON_DIR / "uploads.json"               # list[{user_id, video_id, title, category, uploaded_at_iso}]
USER_LIMITS = JSON_DIR / "user_limits.json"          # {user_id: int}
USER_QUEUES = JSON_DIR / "user_queues.json"          # {user_id: [{url, scheduled_for_iso}]}
PENDING_UPLOADS = JSON_DIR / "pending_uploads.json"  # {msg_id: {...}}

# NEW: prevent re-uploads (source link/signature store)
PROCESSED_SOURCES = JSON_DIR / "processed_sources.json"  # {source_id: {first_seen_iso, uploaded_video_id?}}
SEEN_REELS_FILE = JSON_DIR / "seen_reels.json"           # {shortcode: iso}

DEFAULT_DAILY_LIMIT = 10

YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]

# Timezone
IST = timezone(timedelta(hours=5, minutes=30))  # Asia/Kolkata
import logging

# Base logging setup: hide INFO from libraries
logging.basicConfig(level=logging.WARNING)

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("apscheduler").setLevel(logging.ERROR)
logging.getLogger("telegram").setLevel(logging.ERROR)

# Your own logger (you control what prints)
logger = logging.getLogger("insta2yt")
logger.setLevel(logging.INFO)  # keep your own info/debug visible


# URL validation
YOUTUBE_RE = re.compile(r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/", re.I)
INSTA_RE = re.compile(r"(https?://)?(www\.)?instagram\.com/", re.I)

# Category-specific hashtags
CATEGORY_HASHTAGS = {
    "romantic": [
        "romantic", "love", "couplegoals", "romance", "couple", "lovequotes", "wedding", "kiss", "romanticcouples",
        "hopelessromantic", "romanticdinner", "romanticquotes", "frasesromanticas", "romanticwedding", "relationshipgoals",
        "couplevideos", "longdistancerelationship", "datenightideas", "relationshiptips", "lover", "soulmate", "passion",
        "affection", "hug", "together", "forever", "valentine", "anniversary", "dating", "sweet", "adorable", "heart",
        "romantica", "biromantic", "panromantic", "aromantic", "romantico", "loveisintheair", "romanticvibes", "couplesgoals",
        "lovestory", "truelove", "romanticnight", "candlelight", "roses", "chocolate", "poetry", "serenade", "moonlight",
        "eternallove", "firstlove", "heartbreak", "reunion", "proposal"
    ],
    "cars": [
        "cars", "car", "carsofinstagram", "carporn", "bmw", "carphotography", "auto", "carlifestyle", "automotive", "jdm",
        "carswithoutlimits", "supercars", "ford", "porsche", "luxurycars", "instacars", "exoticcars", "amazingcars247",
        "classiccars", "carshow", "carstagram", "carspotting", "turbo", "racing", "vw", "volkswagen", "carros", "veiculos",
        "seminovos", "chevrolet", "automoveis", "carrosusados", "carlovers", "cargram", "caroftheday", "carlife",
        "carenthusiast", "carpics", "instacar", "speed", "drift", "musclecar", "sportscar", "vintagecar", "electriccar",
        "offroad", "rally", "hypercar", "tuning", "engine", "wheels", "modified", "racecar"
    ],
    "nature": [
        "nature", "naturephotography", "travel", "photooftheday", "instagood", "naturelovers", "landscape", "beautiful",
        "art", "picoftheday", "naturelover", "natureza", "instanature", "mothernature", "nature_perfection", "naturegram",
        "nature_brilliance", "natureaddict", "discover_earthpix", "landscapetreastures", "landscapeoftheday", "landscape_capture",
        "landscapephotos", "landscapes", "wildlife", "outdoors", "hiking", "adventure", "forest", "mountain", "ocean",
        "river", "sunset", "sunrise", "flowers", "trees", "sky", "clouds", "earth", "planetearth", "naturevibes", "green",
        "ecology", "biodiversity", "nationalpark", "trail", "camping", "explore", "wanderlust", "seascape", "wilderness",
        "eco", "conservation"
    ],
    "random": [
        "reels", "reelsinstagram", "instagram", "viral", "trending", "explore", "explorepage", "instagood", "fyp", "love",
        "reelstrending", "funnyreels", "viralreels", "reelsofinstagram", "instagramreels", "reelsindia", "reelsviral",
        "reelsdance", "reelsmusic", "viralmemes", "contentcreator", "trendingreels", "instadaily", "viralvideos", "igaddict",
        "instafamous", "reelvideo", "reelsvideo", "memes", "funny", "humor", "lol", "dance", "music", "fashion", "beauty",
        "food", "travel", "fitness", "motivation", "quotes", "art", "photography", "pets", "cats", "dogs", "baby", "family",
        "friends", "challenge", "diy", "hack", "tips", "lifehacks"
    ]
}

# ====================== Trending tuning ======================
def _ig_session(proxies=None) -> requests.Session:
    s = requests.Session()
    cookies = _load_instagram_cookies_netscape(pathlib.Path("instagram_cookies.txt"))
    if "sessionid" not in cookies or "csrftoken" not in cookies:
        logger.error("instagram_cookies.txt missing required cookies (sessionid/csrftoken).")
        return s  # will fail fast upstream

    s.cookies.update(cookies)
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-IN,en;q=0.9",
        "Referer": "https://www.instagram.com/",
        "X-IG-App-ID": "936619743392459",       # web app id
        "X-Requested-With": "XMLHttpRequest",
    })
    if proxies:
        s.proxies.update(proxies)

    try:
        # warm up: some accounts require a home hit for claim/cap cookies
        s.get("https://www.instagram.com/", timeout=15)
    except Exception as e:
        logger.info("IG warmup failed: %s", e)
    return s


def _parse_explore_sections_from_html(html: str) -> list:
    # try to find "sections":[ ... ] reliably
    m = re.search(r'"sections"\s*:\s*(\[[\s\S]+?\])\s*,\s*"more_available"', html)
    if not m:
        return []
    try:
        return json.loads(m.group(1))
    except Exception:
        return []


def _paginate_tag_sections(s: requests.Session, tag: str, headers: dict, max_pages: int = 3) -> list:
    """Fetch multiple pages of a tag sections feed (web)."""
    out = []
    cursor = None
    for _ in range(max_pages):
        params = {"page": "1"}
        if cursor:
            params["max_id"] = cursor
        r = s.get(f"https://www.instagram.com/api/v1/tags/sections/?tag_name={tag}",
                  headers=headers, params=params, timeout=20)
        if r.status_code != 200: break
        data = r.json()
        sections = data.get("sections") or []
        _gather_from_sections(sections, out)
        cursor = data.get("next_max_id")
        if not cursor: break
    return out



TRENDING_CFG = {
    "hashtags": ["reels", "viral", "trending", "foryou", "explore", "funny", "love", "wow"],
    "min_score": 5_000,
    "max_age_hours": 72,
    "dedupe_days": 7,
    "max_candidates": 100,
    "yt_fallback": False,
    "auto_process_delay_sec": 10,
}

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

# ---------------- Proxy helpers ----------------
@contextmanager
def _no_env_proxy():
    """Temporarily remove env proxies so yt-dlp truly runs with no proxy."""
    keys = ["HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy","ALL_PROXY","all_proxy","NO_PROXY","no_proxy"]
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            if k in os.environ:
                del os.environ[k]
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                if k in os.environ:
                    del os.environ[k]
            else:
                os.environ[k] = v

# ---------------- yt-dlp download (with proxy retry) ----------------
def _build_ydl_opts(url: str, proxy_value: Optional[str]) -> dict:
    opts = {
        "format": "bestvideo*[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(TEMP_DIR / "%(id)s.%(ext)s"),
        "nocheckcertificate": True,
        "noprogress": True,
    }
    # Cookies
    if "instagram.com" in url:
        cookie_file = pathlib.Path("instagram_cookies.txt")
        if cookie_file.exists():
            opts["cookiefile"] = str(cookie_file)
        else:
            logger.warning("Instagram cookies file not found; IG downloads may fail.")
    if ("youtube.com" in url) or ("youtu.be" in url):
        y_cookie = pathlib.Path("youtube_cookies.txt")
        if y_cookie.exists():
            opts["cookiefile"] = str(y_cookie)
    # Proxy
    if proxy_value:
        opts["proxy"] = proxy_value
    return opts

def download_video(url: str) -> pathlib.Path:
    """
    Try with configured proxy (if any). If we hit a proxy error, retry once with NO proxy and env proxies disabled.
    """
    logger.info("Downloading: %s", url)

    def run_dl(ydl_opts, disable_env_proxy=False):
        if disable_env_proxy:
            with _no_env_proxy():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    return pathlib.Path(ydl.prepare_filename(info))
        else:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return pathlib.Path(ydl.prepare_filename(info))

    try:
        return run_dl(_build_ydl_opts(url, PROXY or None))
    except Exception as e:
        err = str(e)
        proxy_signals = ("ProxyError", "proxy", "Tunnel connection failed", "407", "403")
        if any(s in err for s in proxy_signals):
            logger.warning("Retrying download with NO proxy due to proxy error...")
            return run_dl(_build_ydl_opts(url, None), disable_env_proxy=True)
        raise

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
        positions = sorted(set([random.uniform(0.2, max(0.4, duration - 0.2)) for _ in range(num_frames)]))
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

# UPGRADED PROMPT
GEMINI_PROMPT = """
You are an expert YouTube Shorts metadata strategist.
Given 5 video frames, create a STRICT JSON object with keys:

- category: one of ["Romantic","Cars","Nature","Random"] (exactly one)
- title: 58-72 chars, high CTR, no emojis, no quotes, capitalize major words
- description: 130-200 words, includes a 1-2 line hook, context, and a simple call-to-action
- tags: 12-18 plain single-word tags (no #), lowercase, diverse but relevant
- hashtags: 12-18 hashtags starting with #, mix of broad (#shorts) and niche; avoid duplicates

Rules:
- Keep it brand-safe; no profanity or sensitive claims.
- Prefer powerful action verbs and concrete nouns.
- Don't repeat the same word in title more than once.
- No trailing punctuation in the title.
- Return ONLY the JSON (no backticks, no prose).
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
        "generationConfig": {"temperature": 0.35, "maxOutputTokens": 900}
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
    hashtags = [f"#{t}" for t in ["shorts","viral","trending","explore","reels","fyp","youtube","growth"]]
    return {"category": category, "title": title, "description": description, "tags": tags, "hashtags": hashtags}

def normalize_meta(obj: Optional[Dict[str, Any]], hint: str) -> Dict[str, Any]:
    if not obj:
        return fallback_metadata(hint)
    category = str(obj.get("category") or "Random").strip().title()
    if category not in {"Romantic","Cars","Nature","Random"}:
        category = "Random"
    title = str(obj.get("title") or hint or "Untitled Short").strip()[:100]
    title = title.rstrip(".!?")
    description = str(obj.get("description") or "").strip()
    tags = obj.get("tags") or []
    hashtags = obj.get("hashtags") or []
    tags = [re.sub(r"^#","", str(t)).strip().lower() for t in tags if str(t).strip()]
    hashtags = [h if str(h).startswith("#") else f"#{str(h).strip()}" for h in hashtags if str(h).strip()]
    # de-dup hashtags
    seen=set(); hashtags=[h for h in hashtags if not (h in seen or seen.add(h))]
    return {"category": category, "title": title, "description": description, "tags": tags[:18], "hashtags": hashtags[:18]}

# ---------------- YouTube helpers ----------------
def load_client_secrets(category_key: str):
    mapping = {
        "romantic": JSON_DIR / "romantic_client_secrets.json",
        "cars": JSON_DIR / "cars_client_secrets.json",
        "nature": JSON_DIR / "nature_client_secrets.json",
        "random": JSON_DIR / "random_client_secrets.json",
    }
    path = mapping.get(category_key.lower())
    if path and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    elif DEFAULT_CLIENT_SECRETS.exists():
        logger.warning("Client secrets for %s not found. Using fallback: %s", category_key, DEFAULT_CLIENT_SECRETS)
        with DEFAULT_CLIENT_SECRETS.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Client secrets file for category {category_key} not found at {path} and no fallback present.")

def _token_file(category_key: str) -> str:
    return str(PICKLES_DIR / f"token_{category_key.lower()}.pickle")

def get_youtube_service(client_secrets, category_key: str):
    token_file = _token_file(category_key)
    creds = None
    if os.path.exists(token_file):
        with open(token_file, "rb") as f:
            creds = pickle.load(f)

    if not creds or not getattr(creds, "valid", False):
        if creds and creds.expired and getattr(creds, "refresh_token", None):
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds:
            flow = InstalledAppFlow.from_client_config(client_secrets, YOUTUBE_SCOPES)
            try:
                creds = flow.run_local_server(port=0, prompt='consent')
            except Exception:
                print("\nNo local browser available. Using console OAuth.")
                print("Open the URL shown below in any browser, then paste the code here.\n")
                creds = flow.run_console()

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
        return text.split()[0] in {"/report","/ping","/stats","/setlimt","/find","/start"}
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

# ---------------- Re-upload prevention helpers ----------------
def _read_processed_sources() -> Dict[str, Dict[str, Any]]:
    d = _read_json(PROCESSED_SOURCES, {})
    return d if isinstance(d, dict) else {}

def _write_processed_sources(d: Dict[str, Dict[str, Any]]):
    _write_json(PROCESSED_SOURCES, d)

def _extract_source_id(url: str) -> str:
    url = url.strip()
    m = re.search(r"instagram\.com/(?:reel|reels)/([A-Za-z0-9_-]+)", url)
    if m:
        return f"ig:{m.group(1)}"
    # youtube watch / shorts / youtu.be
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{6,})", url)
    if m:
        return f"yt:{m.group(1)}"
    m = re.search(r"youtube\.com/(?:shorts/|watch\?v=)([A-Za-z0-9_-]{6,})", url)
    if m:
        return f"yt:{m.group(1)}"
    # fallback: normalized url without query
    base = re.split(r"[?#]", url)[0]
    return f"url:{base.lower()}"

def _is_already_processed(url: str) -> bool:
    sid = _extract_source_id(url)
    d = _read_processed_sources()
    return sid in d

def _mark_processed_source(url: str, uploaded_video_id: Optional[str] = None):
    sid = _extract_source_id(url)
    d = _read_processed_sources()
    d[sid] = {"first_seen_iso": datetime.now(IST).isoformat()}
    if uploaded_video_id:
        d[sid]["uploaded_video_id"] = uploaded_video_id
    _write_processed_sources(d)

# ---------------- Category balance helpers ----------------
def _last_n_categories(user_id: int, n: int = 4) -> List[str]:
    logs = _read_json(UPLOAD_LOG, [])
    cats = []
    for row in reversed(logs):
        if int(row.get("user_id", 0)) == int(user_id):
            cats.append(str(row.get("category","")).title())
            if len(cats) >= n:
                break
    return cats

# ---------------- Category midnight helpers ----------------
def _today_category_counts(user_id: int) -> Dict[str, int]:
    """
    Count how many uploads per category the user has TODAY (IST).
    Resets naturally at midnight IST.
    """
    logs = _read_json(UPLOAD_LOG, [])
    today = datetime.now(IST).date()
    counts: Dict[str, int] = {}
    for row in logs:
        try:
            t = datetime.fromisoformat(row["uploaded_at_iso"]).astimezone(IST).date()
        except Exception:
            continue
        if int(row.get("user_id", 0)) != int(user_id):
            continue
        if t != today:
            continue
        cat = str(row.get("category", "")).title()
        counts[cat] = counts.get(cat, 0) + 1
    return counts

# ---------------- Fancy progress animator (fake) ----------------
class _ProgressAnimator:
    FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

    def __init__(self, msg: Message, label: str):
        self.msg = msg
        self.label = label
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        p = 0
        i = 0
        while not self._stop.is_set():
            p = min(99, p + random.randint(1, 3))
            bars = int(p / 10)
            bar = "█"*bars + "▒"*(10-bars)
            try:
                await self.msg.edit_text(f"{self.label} {self.FRAMES[i%len(self.FRAMES)]}  [{bar}] {p}%")
            except Exception:
                pass
            i += 1
            await asyncio.sleep(0.8)

    async def stop(self, final_text: Optional[str] = None):
        self._stop.set()
        try:
            await asyncio.sleep(0)  # let loop end
            if final_text:
                await self.msg.edit_text(final_text)
        except Exception:
            pass

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
            lines.append(f"  - {it['title']}[](https://youtu.be/{it['video_id']})")
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

# ---------------- /start (interactive) ----------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = [
        [InlineKeyboardButton("🔎 Find Trending Now", callback_data="start_find")],
        [InlineKeyboardButton("📊 Stats", callback_data="start_stats"),
         InlineKeyboardButton("🗓️ Weekly Report", callback_data="start_report")],
        [InlineKeyboardButton("⚙️ Limit 5", callback_data="setlimt::5"),
         InlineKeyboardButton("⚙️ Limit 10", callback_data="setlimt::10"),
         InlineKeyboardButton("⚙️ Limit 20", callback_data="setlimt::20")],
    ]
    if GITHUB_URL:
        rows.append([InlineKeyboardButton("🐙 Visit GitHub", url=GITHUB_URL)])
    keyboard = InlineKeyboardMarkup(rows)
    text = (
        "👋 Welcome to Insta2YT!\n\n"
        "Paste an Instagram or YouTube link and I’ll process it.\n\n"
        "Commands:\n"
        "• /ping – Check if I’m alive\n"
        "• /report – Weekly upload report\n"
        "• /stats – View recent stats\n"
        "• /setlimt <n> – Set daily upload limit\n"
        "• /find – Force-search a trending reel now\n\n"
        "🔥 I also auto-pick a trending reel every 4 hours."
    )
    await update.message.reply_text(text, reply_markup=keyboard)

# Handler for button clicks
async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = (query.data or "")
    chat_id = query.message.chat.id

    if data.startswith("cat::") or data == "cancel_upload":
        if data == "cancel_upload":
            pend = pop_pending(query.message.message_id)
            if pend:
                try:
                    p = pathlib.Path(pend["path"])
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            await query.edit_message_text("❎ Canceled.")
            return

        if not data.startswith("cat::"):
            return
        # callback_data = "cat::<Category>::<filename>"
        parts = data.split("::", 2)
        new_cat = parts[1] if len(parts) >= 2 else "Random"
        pend = get_pending(query.message.message_id)
        if not pend:
            await query.edit_message_text("No pending upload.")
            return
        pend["category"] = new_cat
        save_pending(query.message.message_id, pend)
        await query.edit_message_text(f"Category set to *{new_cat}*. Uploading now...", parse_mode="Markdown")
        await proceed_upload(context, query.message.chat.id, query.message.message_id)
        return

    # Reel and start menu buttons
    if data.startswith("go::"):
        url = data.split("::", 1)[1]
        pending_reels.pop(query.message.message_id, None)
        await query.edit_message_text(f"🚀 Processing reel: {url}")
        await process_url_for_chat(context, chat_id, url, enforce_balance=True, from_auto_find=True)
        return

    if data == "find_another":
        pending_reels.pop(query.message.message_id, None)
        await query.edit_message_text("🔄 Searching another reel…")
        url = get_trending_reel_url()
        if url:
            await present_reel(context, chat_id, url)
        else:
            await context.bot.send_message(chat_id, "⚠️ Could not fetch another reel.")
        return

    if data == "start_find":
        await query.message.reply_text("/find")
        return

    if data == "start_stats":
        await query.message.reply_text("/stats")
        return

    if data == "start_report":
        await query.message.reply_text("/report")
        return

    if data.startswith("setlimt::"):
        try:
            n = int(data.split("::", 1)[1])
            await query.message.reply_text(f"/setlimt {n}")
        except Exception as e:
            await query.edit_message_text(f"❌ Failed to set limit: {e}")
        return


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
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
        if text.startswith("/find"):
            return await find(update, context)
        return

    # It's a URL
    user_id = update.effective_chat.id
    url = text

    if not (YOUTUBE_RE.search(url) or INSTA_RE.search(url)):
        return

    # Re-upload prevention (for pasted links)
    if _is_already_processed(url):
        await update.message.reply_text("🚫 This link looks already processed earlier. Skipping.")
        return

    if not can_process_now(user_id):
        enqueue_for_tomorrow_noon(user_id, url)
        limit = get_user_limit(user_id)
        await update.message.reply_text(f"⏸️ Daily limit {limit} reached. Queued for tomorrow 12:00 PM IST.")
        return

    await process_url(update, context, url)

async def process_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str,
                      enforce_balance: bool = False, from_auto_find: bool = False, from_auto_find_retry_count: int = 0):
    user_id = update.effective_chat.id

    # Duplicate check again (for auto-find / present)
    if _is_already_processed(url):
        if from_auto_find:
            await context.bot.send_message(user_id, "🚫 That reel was already processed. Finding another…")
            url2 = get_trending_reel_url()
            if url2:
                await present_reel(context, user_id, url2)
            else:
                await context.bot.send_message(user_id, "⚠️ Could not fetch another reel right now.")
        else:
            await update.message.reply_text("🚫 This link looks already processed earlier. Skipping.")
        return

    # Animated downloading
    msg = await update.message.reply_text("⬇️ Downloading…")
    dl_anim = _ProgressAnimator(msg, "⬇️ Downloading")

    try:
        downloaded = download_video(url)
    except Exception as e:
        await dl_anim.stop(f"❌ Download failed: {e}")
        await update.message.reply_text(
            "Tip: set PROXY in code if you're behind a proxy, or add youtube_cookies.txt / instagram_cookies.txt."
        )
        return

    await dl_anim.stop("🔍 Analyzing…")
    md_anim = _ProgressAnimator(msg, "🧠 Generating metadata")

    try:
        images_bytes = extract_frames(downloaded, num_frames=5)
        meta_raw = call_gemini(images_bytes)
        meta = normalize_meta(meta_raw, downloaded.stem)
    except Exception as e:
        logger.warning("Metadata generation failed: %s", e)
        meta = fallback_metadata(downloaded.stem)

    await md_anim.stop()

    # Category balance: only for auto-find / /find flows (per-day cap)
    if enforce_balance:
        today_counts = _today_category_counts(user_id)
        cat_title = meta["category"].title()
        if today_counts.get(cat_title, 0) >= 4:
            # Already uploaded 4 of this category today → find another
            available_cats = [c for c in ["romantic", "cars", "nature", "random"] if today_counts.get(c.title(), 0) < 4]
            if not available_cats or from_auto_find_retry_count >= 2:
                await context.bot.send_message(user_id, "⚠️ Category limits reached. Skipping this auto find.")
                try:
                    if downloaded.exists():
                        downloaded.unlink()
                except Exception:
                    pass
                return
            other_cat = random.choice(available_cats)
            try:
                if downloaded.exists():
                    downloaded.unlink()
            except Exception:
                pass
            await context.bot.send_message(
                user_id,
                f"🧮 Already uploaded 4 *{cat_title}* reels today. Finding a different category ({other_cat.title()})…",
                parse_mode="Markdown"
            )
            url2 = get_trending_reel_url(target_hashtags=CATEGORY_HASHTAGS[other_cat])
            if url2:
                await process_url(update, context, url2, enforce_balance=True, from_auto_find=from_auto_find, from_auto_find_retry_count=from_auto_find_retry_count + 1)
            else:
                await context.bot.send_message(user_id, "⚠️ Could not fetch another reel right now.")
            return

    if from_auto_find:
        # No category prompt for auto, directly upload with proposed category
        category_key = meta["category"].lower()
        if category_key not in {"romantic", "cars", "nature", "random"}:
            category_key = "random"
        mark_processed(user_id)
        upload_msg = await context.bot.send_message(user_id, "🚀 Uploading auto reel…")
        try:
            loop = asyncio.get_running_loop()
            vid = upload_youtube(downloaded, meta, category_key, upload_msg, context, loop)
            await context.bot.send_message(
                user_id,
                f"🎉 Auto uploaded!\n📌 Title: {meta['title']}\n🏷️ Category: {category_key.title()}\n🔗 https://youtu.be/{vid}"
            )
            add_upload_log(user_id, vid, meta["title"], category_key.title())
            _mark_processed_source(url, uploaded_video_id=vid)
        except Exception as e:
            await context.bot.send_message(user_id, f"❌ Auto upload failed: {e}")
        finally:
            try:
                if downloaded.exists():
                    downloaded.unlink()
            except Exception:
                pass
        return

    # Manual: Category correction buttons; auto proceed after 60s
    categories = ["Romantic","Cars","Nature","Random"]
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(("✅ " if cat == meta["category"] else "") + cat, callback_data=f"cat::{cat}::{downloaded.name}")]
        for cat in categories
    ] + [
        [InlineKeyboardButton("🚫 Cancel", callback_data="cancel_upload")]
    ])
    prompt = (
        f"📂 Proposed category: *{meta['category']}*\n"
        f"📝 Title: {meta['title']}\n\n"
        "You can tweak the category below. Upload starts automatically in 60 seconds."
    )
    sent = await update.message.reply_text(prompt, reply_markup=keyboard, parse_mode="Markdown")

    pending = {
        "user_id": user_id,
        "path": str(downloaded),
        "meta": meta,
        "category": meta["category"],
        "expires_at_iso": (datetime.now(IST) + timedelta(seconds=60)).isoformat(),
        "source_msg_id": sent.message_id,
        "source_url": url,  # keep original for de-dupe marking
    }
    save_pending(sent.message_id, pending)

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
    source_url = obj.get("source_url")
    category_key = (obj.get("category") or meta.get("category") or "Random").lower()
    if category_key not in {"romantic","cars","nature","random"}:
        category_key = "random"

    mark_processed(user_id)

    try:
        msg = await context.bot.send_message(chat_id, "🚀 Uploading… preparing")
    except Exception:
        return

    try:
        vid = upload_youtube(video_path, meta, category_key, msg, context, loop)
        await context.bot.send_message(
            chat_id,
            f"🎉 Uploaded!\n📌 Title: {meta['title']}\n🏷️ Category: {category_key.title()}\n🔗 https://youtu.be/{vid}"
        )
        add_upload_log(user_id, vid, meta["title"], category_key.title())
        if source_url:
            _mark_processed_source(source_url, uploaded_video_id=vid)
    except Exception as e:
        await context.bot.send_message(chat_id, f"❌ Upload failed: {e}")
        return
    finally:
        try:
            if video_path.exists():
                video_path.unlink()
        except Exception:
            pass

# ---------------- JobQueue: process due queued items ----------------
async def process_url_for_chat(context: ContextTypes.DEFAULT_TYPE, chat_id: int, url: str,
                               enforce_balance: bool = False, from_auto_find: bool = False, from_auto_find_retry_count: int = 0):
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
    await process_url(tmp_update, context, url, enforce_balance=enforce_balance, from_auto_find=from_auto_find, from_auto_find_retry_count=from_auto_find_retry_count)

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

# ---------------- Trending: cookies + scoring + multi-source ----------------
def _load_instagram_cookies_netscape(path: pathlib.Path) -> Dict[str, str]:
    jar: Dict[str, str] = {}
    if not path.exists():
        return jar
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip() or raw.startswith("#"):
                continue
            parts = raw.strip().split("\t")
            if len(parts) >= 7:
                name, value = parts[5], parts[6]
                jar[name] = value
    return jar

def _extract_reel_urls_from_html(html: str) -> List[str]:
    """
    Extract reel URLs from the HTML response (looking for Instagram reel shortcodes).
    """
    urls = re.findall(r'https://www\.instagram\.com/reel/[A-Za-z0-9_-]+/', html)
    return list(set(urls))  # Remove duplicates


def _load_seen_reels() -> Dict[str, str]:
    data = _read_json(SEEN_REELS_FILE, {})
    return data if isinstance(data, dict) else {}

def _save_seen_reels(d: Dict[str, str]):
    _write_json(SEEN_REELS_FILE, d)

def _remember_shortcode(shortcode: str):
    try:
        d = _load_seen_reels()
        d[shortcode] = datetime.now(IST).isoformat()
        _save_seen_reels(d)
    except Exception:
        pass

def _seen_recent(shortcode: str, dedupe_days: int) -> bool:
    try:
        d = _load_seen_reels()
        ts = d.get(shortcode)
        if not ts:
            return False
        then = datetime.fromisoformat(ts).astimezone(IST)
        return datetime.now(IST) - then < timedelta(days=dedupe_days)
    except Exception:
        return False

def _score_media(m: Dict[str, Any], now_ts: float) -> tuple:
    sc = m.get("code") or m.get("shortcode")
    if not sc:
        return (0.0, None, None)
    plays = int(m.get("play_count") or m.get("view_count") or 0)
    likes = int(m.get("like_count") or 0)
    comments = int(m.get("comment_count") or 0)

    taken_at = m.get("taken_at") or m.get("device_timestamp") or 0
    age_h = None
    if taken_at:
        try:
            age_h = max(1.0, (now_ts - float(taken_at)) / 3600.0)
        except Exception:
            age_h = None

    # per-hour rates; guard divide-by-zero
    if age_h and age_h > 0:
        vph = plays / age_h
        lph = likes / age_h
        cph = comments / age_h
    else:
        vph = plays
        lph = likes
        cph = comments

    score = vph + 0.3 * lph + 0.2 * cph
    # small recency bump under 24h
    if age_h is not None:
        if age_h < 12:  score *= 2.2
        elif age_h < 24: score *= 1.6
        elif age_h < 48: score *= 1.15
    return (score, sc, age_h)


def _gather_from_sections(sections: list, out: List[Dict[str, Any]]):
    for sec in sections or []:
        layout = (sec or {}).get("layout_content", {})
        for wrap in layout.get("medias", []):
            m = (wrap or {}).get("media", {})
            if not (m.get("product_type") in ("clips", "reel") or m.get("is_reel_media")):
                continue
            out.append(m)

def _pick_best_candidate(candidates: List[Dict[str, Any]]) -> Optional[str]:
    if not candidates:
        return None
    now_ts = datetime.now(IST).timestamp()
    scored = []
    for m in candidates[: TRENDING_CFG["max_candidates"]]:
        score, sc, age_h = _score_media(m, now_ts)
        if not sc or score < TRENDING_CFG["min_score"]:
            continue
        if age_h is not None and age_h > TRENDING_CFG["max_age_hours"]:
            continue
        if _seen_recent(sc, TRENDING_CFG["dedupe_days"]):
            continue
        # prevent re-upload based on processed_sources
        if _is_already_processed(f"https://www.instagram.com/reel/{sc}/"):
            continue
        scored.append((score, sc))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    pool = [u for _, u in scored[:10]] if len(scored) > 10 else [u for _, u in scored]
    sc = random.choice(pool)
    url = f"https://www.instagram.com/reel/{sc}/"
    try:
        cookies = _load_instagram_cookies_netscape(pathlib.Path("instagram_cookies.txt"))
        s = requests.Session(); s.cookies.update(cookies)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Referer": "https://www.instagram.com/"}
        s.get(url, headers=headers, timeout=10, allow_redirects=True)
        _remember_shortcode(sc)
        return url
    except Exception:
        return url

def _yt_shorts_fallback() -> Optional[str]:
    try:
        r = requests.get("https://www.youtube.com/feed/trending", timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None
        ids = re.findall(r'https://www\.youtube\.com/shorts/([A-Za-z0-9_-]{8,})', r.text)
        if not ids:
            return None
        vid = random.choice(list(dict.fromkeys(ids)))
        return f"https://www.youtube.com/shorts/{vid}"
    except Exception:
        return None
def get_trending_reel_url(target_hashtags: Optional[List[str]] = None) -> Optional[str]:
    try:
        proxies = {"http": PROXY, "https": PROXY} if PROXY else None
        s = _ig_session(proxies=proxies)

        # Validate cookies early
        if "sessionid" not in s.cookies:
            logger.error("instagram_cookies.txt missing 'sessionid'. Export fresh cookies while logged in.")
            return None

        headers = dict(s.headers)  # already has UA/app-id/etc.

        candidates: List[Dict[str, Any]] = []

        # 1) Explore JSON (same as you had)
        try:
            api_url = ("https://www.instagram.com/api/v1/discover/web/explore_grid/"
                       "?is_prefetch=false&omit_cover_media=true&use_sectional_payload=true&include_fixed_destinations=true")
            r = s.get(api_url, headers=headers, timeout=20)
            if r.status_code == 200 and "application/json" in r.headers.get("Content-Type", ""):
                data = r.json()
                _gather_from_sections(data.get("sections", []), candidates)
        except Exception as ex:
            logger.info("Explore JSON error: %s", ex)

        # 2) Explore HTML (prefer your own Explore with cookies)
        if len(candidates) < 20:
            try:
                r2 = s.get("https://www.instagram.com/explore/", headers=headers, timeout=20)
                if r2.status_code == 200:
                    sections = _parse_explore_sections_from_html(r2.text)
                    _gather_from_sections(sections, candidates)
                    # last-ditch: direct reel links in html
                    if len(candidates) < 10:
                        urls = _extract_reel_urls_from_html(r2.text)
                        if urls:
                            return random.choice(urls)
            except Exception as ex:
                logger.info("Explore HTML error: %s", ex)

        # 3) Hashtag-focused with pagination (use target or default)
        if len(candidates) < 30:
            if target_hashtags is None:
                target_hashtags = TRENDING_CFG["hashtags"] + ["india","bollywood","hindi","mumbai","desi","foryou","comedy","india_reels"]
            for tag in target_hashtags:
                try:
                    # page 0 info
                    rt = s.get(f"https://www.instagram.com/api/v1/tags/web_info/?tag_name={tag}",
                               headers=headers, timeout=20)
                    if rt.status_code == 200 and "application/json" in rt.headers.get("Content-Type",""):
                        data = rt.json()
                        sections = data.get("data", {}).get("top", {}).get("sections") or data.get("sections")
                        _gather_from_sections(sections or [], candidates)
                    # extra pages
                    more = _paginate_tag_sections(s, tag, headers, max_pages=2)
                    if more:
                        candidates.extend(more)
                except Exception:
                    pass
                if len(candidates) >= TRENDING_CFG["max_candidates"]:
                    break

        # 4) Pick best by per-hour score + your dedupe/age filters
        url = _pick_best_candidate(candidates)
        if url:
            return url

        # 5) Optional: YT shorts fallback (toggle in TRENDING_CFG)
        if TRENDING_CFG.get("yt_fallback"):
            return _yt_shorts_fallback()

        return None

    except Exception as e:
        logger.error("get_trending_reel_url fatal: %s", e)
        return None


# ---------------- Present found reel with buttons + auto-process ----------------
pending_reels: Dict[int, str] = {}  # {message_id: url}

async def present_reel(context: ContextTypes.DEFAULT_TYPE, chat_id: int, reel_url: str):
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Go", callback_data=f"go::{reel_url}")],
        [InlineKeyboardButton("🔄 Find Another", callback_data="find_another")]
    ])
    sent = await context.bot.send_message(chat_id, f"🔥 Trending Reel Found:\n{reel_url}", reply_markup=keyboard)
    pending_reels[sent.message_id] = reel_url

    async def auto_process():
        await asyncio.sleep(TRENDING_CFG.get("auto_process_delay_sec", 10))
        if sent.message_id in pending_reels:
            url = pending_reels.pop(sent.message_id)
            await context.bot.send_message(chat_id, "⏳ No response, auto-processing reel...")
            await process_url_for_chat(context, chat_id, url, enforce_balance=True, from_auto_find=True)

    asyncio.create_task(auto_process())

# ---------------- /find command (manual trigger; auto job stays on) ----------------
async def find(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    # Send an initial message indicating the bot is searching for a reel
    searching_message = await context.bot.send_message(
        chat_id,
        "🔍 Searching for a trending reel... Please wait! 👀"
    )

    # Add an animated scrolling progress to the message
    scroll_anim = _ProgressAnimator(searching_message, "🔄 Searching reels: ")

    # Fetch the trending reel URL using the optimized function
    url = get_trending_reel_url()

    # Stop the animation after finding the reel or a timeout
    await scroll_anim.stop()

    if url:
        # Send the found reel link after the animation stops
        await context.bot.send_message(chat_id, f"🎉 Trending Reel found! Watch it here: {url}")
        # Trigger the processing and uploading of the found reel
        await process_url(update, context, url, enforce_balance=True)  # Trigger the upload process with balance
    else:
        # If no reel is found, inform the user
        await context.bot.send_message(chat_id, "⚠️ Could not fetch a trending reel right now. Please try again later.")


# ---------------- 4-hour auto trending job ----------------
async def trending_reel_job(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id  # define first

    logger.info("⏰ 4-hour auto job triggered")
    await context.bot.send_message(chat_id, "⏰ 4-hour auto job triggered")

    url = get_trending_reel_url()
    if url:
        await present_reel(context, chat_id, url)
    else:
        await context.bot.send_message(chat_id, "⚠️ No trending reels found this time.")


# ---------------- Main ----------------
def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in code.")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Commands (original + new)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("report", report))
    app.add_handler(CommandHandler("setlimt", setlimt))
    app.add_handler(CommandHandler("find", find))

    # Unified callback query handler
    app.add_handler(CallbackQueryHandler(callback_query_handler))

    # Only react to texts that are allowed (URLs). Everything else is ignored. (original)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # background job to check queued URLs (original)
    app.job_queue.run_repeating(queue_watcher_job, interval=60, first=10)

    # background job: auto trending reel every 4 hours (NEW)
    if OWNER_CHAT_ID:
        app.job_queue.run_repeating(trending_reel_job, interval=4*60*60, first=30, chat_id=OWNER_CHAT_ID)

    logger.info("Bot starting...")
    app.run_polling()

if __name__ == "__main__":
    main()
