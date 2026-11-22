"""Configuration management."""
import json
import os

DEFAULT_CONFIG = {
    "port": 8081,
    "host": "0.0.0.0",
    "retry_attempts": 3,
    "retry_delay_sec": 2,
    "request_timeout_sec": 180,
    "ticket_timeout_sec": 25,
    "gemini_bl": "boq_gemini-web-uiserver_20261122.06_p0",
    "auth_user": None,
    "xsrf_token": None,
    "default_model": "gemini-3.7-flash",
    "log_requests": True,
    "cookie_file": None,
    "proxy": None,
    "api_keys": [],
    "temporary_chats": False,
    "send_model_ticket": True,
    "model_tickets": {},
}

CONFIG = dict(DEFAULT_CONFIG)


def load_config(path: str = None):
    """Load config from JSON file."""
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            CONFIG.update(json.load(f))
    return CONFIG


def find_config():
    """Search for config file in standard locations."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        "./config.json",
        os.path.join(base_dir, "config.json"),
        os.path.expanduser("~/.config/gemini-web2api/config.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None
