# Base44 Dev Environment

## What this app is
A Streamlit chat app (`li2.py`) that streams responses from OpenRouter (OpenAI-compatible API) using the `google/gemini-2.5-flash` model. Despite the README mentioning a CLI tool, the actual app is a Streamlit web UI.

## Running it
```bash
docker compose -f docker-compose.base44.yml up -d
```
- App listens on port 8501 inside the container, mapped to host port 3000.
- Health check: `GET /_stcore/health` returns 200.
- Dependencies (streamlit, openai, python-dotenv) are installed on container startup from `requirements.txt`.

## Required secret
- `OPENROUTER_API_KEY` — OpenRouter API key (get from https://openrouter.ai/keys). Without a valid key the app boots but shows an error and stops. A development placeholder is generated so the container starts; replace it with a real key for the chat to work.

## Notes
- Streamlit is configured with CORS and XSRF protection disabled, headless mode, and polling file watcher (needed for bind-mount live reload).
- Source is bind-mounted at `/app`; edits to `li2.py` hot-reload automatically.
