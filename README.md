
Ista2Shorts Bot 🎥🤖

A **Telegram bot** that downloads videos from **Instagram** and **YouTube**, generates metadata using **Gemini AI**, and uploads them to **YouTube Shorts** automatically.  
<center><img src="https://github.com/user-attachments/assets/1b86cccd-1e2f-4b72-a0c5-3a1f902035ad" hight="350"></center>

It’s designed to help automate content repurposing, SEO optimize uploads, and manage daily upload limits with queuing.

---

## ✨ Features

- **Strict command handling** → responds only to:
  - YouTube/Instagram links
  - `/ping`
  - `/report`
  - `/stats`
  - `/setlimt`
- **Smart video download**:
  - Uses `yt-dlp`
  - Cookie support (`youtube_cookies.txt`, `instagram_cookies.txt`)
  - Proxy retry logic (no-proxy fallback)
- **Metadata generation with Gemini**:
  - Extracts 5 random frames
  - Generates SEO-friendly title, description, tags, hashtags
  - Categories: `Romantic`, `Cars`, `Nature`, `Random`
- **Category confirmation**:
  - Telegram buttons to adjust category
  - If user clicks → upload starts immediately
  - If no response → auto-uploads after 60s
- **Daily upload management**:
  - Per-user daily limit (default: 10, configurable with `/setlimt`)
  - If limit reached → link is queued for **12 PM IST next day**
- **Reporting & stats**:
  - `/report` → shows videos uploaded this week by category
  - `/stats` → lists titles, views, categories (live YouTube stats)
- **Organized file storage**:
  - `Temp/` → downloaded videos (auto-cleaned after upload)
  - `Pickles/` → YouTube OAuth tokens
  - `Json/` → logs, queues, limits, pending uploads

---

## 📂 Project Structure

```bash
insta2shorts-bot/
├── insta2shorts_mvp.py        # main bot script
├── requirements.txt           # dependencies
├── README.md                  # this file
├── Temp/                      # temporary downloads
├── Pickles/                   # OAuth tokens
├── Json/                      # persisted data
├── romantic_client_secrets.json  # you add
├── cars_client_secrets.json      # you add
├── nature_client_secrets.json    # you add
├── random_client_secrets.json    # you add
├── instagram_cookies.txt         # optional
└── youtube_cookies.txt           # optional
````

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/insta2shorts-bot.git
cd insta2shorts-bot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Open `insta2shorts_mvp.py` and replace:

```python
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

### 4. Add YouTube OAuth secrets

Place the following in the root:

* `romantic_client_secrets.json`
* `cars_client_secrets.json`
* `nature_client_secrets.json`
* `random_client_secrets.json`

You can generate them from [Google Cloud Console](https://console.cloud.google.com/).

### 5. Optional: Add cookies

* `youtube_cookies.txt` → for age/region-restricted videos
* `instagram_cookies.txt` → for private/limited IG videos

### 6. Run the bot

```bash
python insta2shorts_mvp.py
```

---

## 📖 Usage

Send the bot:

* A **YouTube** or **Instagram** link → it downloads, generates metadata, asks category, uploads
* `/ping` → check if bot is alive
* `/report` → see what was uploaded this week
* `/stats` → get video titles, views, and categories
* `/setlimt 5` → set your daily limit (default: 10)

Example flow:

1. Send a YT link
2. Bot replies with “Proposed category: Nature …” and category buttons
3. Click “Cars” → bot uploads immediately
4. If no click → bot uploads after 60s using Gemini’s suggestion

---

## 🛠️ Tech Stack

* [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) (Telegram API)
* [yt-dlp](https://github.com/yt-dlp/yt-dlp) (video download)
* [OpenCV](https://opencv.org/) (frame extraction)
* [Google API Client](https://github.com/googleapis/google-api-python-client) (YouTube uploads/stats)
* [Gemini API](https://ai.google.dev/) (metadata generation)

---

## ⚙️ Configuration Options

Inside `insta2shorts_mvp.py`:

* `PROXY` → set to `"http://user:pass@host:port"` if your network requires proxy
* Daily limit per user (default 10) → can be overridden with `/setlimt <n>`
* Categories supported: `Romantic`, `Cars`, `Nature`, `Random`

---

## 🗄️ Data Persistence

* Uploads log → `Json/uploads.json`
* User limits → `Json/user_limits.json`
* Queued links → `Json/user_queues.json`
* Pending uploads (60s confirm window) → `Json/pending_uploads.json`

These files ensure the bot can restart without losing state.

---

## 🔒 Security Notes

* **Do not commit your real tokens** (`TELEGRAM_BOT_TOKEN`, `GEMINI_API_KEY`, cookies, or OAuth secrets) to GitHub.
* Rotate tokens if they’ve been exposed.
* Keep `Pickles/` and `Json/` out of public repos if possible. Add them to `.gitignore`.

---

## 🐛 Troubleshooting

* **YouTube proxy error**

  ```
  ERROR: Unable to connect to proxy ... Tunnel connection failed: 403
  ```

  → Leave `PROXY=""` to force direct connection
  → Or provide a working proxy in code
  → Or add `youtube_cookies.txt`

* **Instagram private videos** → require `instagram_cookies.txt`

* **Upload fails with "youtubeSignupRequired"** → You need to create a YouTube channel for the account.

---

## 🤝 Contributing

Pull requests are welcome!
Ideas you could add:

* More categories (Gaming, Comedy, Travel, etc.)
* Scheduled uploads at custom times
* Thumbnail selection with Gemini
* Admin-only commands

---

## 📜 License

MIT License – free to use, modify, and distribute.

---

## ⭐ Acknowledgements

* [yt-dlp](https://github.com/yt-dlp/yt-dlp)
* [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
* [Google API Client](https://github.com/googleapis/google-api-python-client)
* [Gemini AI](https://ai.google.dev/)

---
