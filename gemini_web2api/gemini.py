"""Gemini StreamGenerate protocol implementation with httpx streaming."""
import json
import time
import uuid
import re
import urllib.request
import urllib.parse
import ssl
import os
import hashlib

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .config import CONFIG

_ssl_ctx = None
_cookie_cache = {"str": "", "sapisid": None, "mtime": 0}
_httpx_client = None


def log(msg: str):
    if CONFIG["log_requests"]:
        import sys
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()


def _get_ssl_ctx():
    global _ssl_ctx
    if _ssl_ctx is None:
        _ssl_ctx = ssl.create_default_context()
    return _ssl_ctx


def _get_httpx_client():
    global _httpx_client
    if _httpx_client is None and HAS_HTTPX:
        proxy = CONFIG.get("proxy")
        transport = httpx.HTTPTransport(proxy=proxy) if proxy else None
        _httpx_client = httpx.Client(transport=transport, timeout=CONFIG["request_timeout_sec"], verify=True)
    return _httpx_client


def load_cookie() -> tuple:
    """Load cookie from file with mtime-based caching."""
    cookie_file = CONFIG.get("cookie_file")
    if not cookie_file or not os.path.exists(cookie_file):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            cookie_file,
            os.path.join(base_dir, "cookie.json"),
            os.path.join(base_dir, "cookie.txt"),
            "./cookie.json",
            "cookie.json",
            os.path.expanduser("~/.config/gemini-web2api/cookie.json")
        ]
        found = False
        for c in candidates:
            if c and os.path.exists(c):
                cookie_file = c
                CONFIG["cookie_file"] = c
                found = True
                break
        if not found:
            return "", None
    try:
        mtime = os.path.getmtime(cookie_file)
        if mtime == _cookie_cache["mtime"] and _cookie_cache["str"]:
            return _cookie_cache["str"], _cookie_cache["sapisid"]
        with open(cookie_file, "r") as f:
            content = f.read().strip()
        if content.startswith("{"):
            data = json.loads(content)
            cookie_str = data.get("cookie", "")
            sapisid = data.get("sapisid", "")
            if not CONFIG.get("xsrf_token") and data.get("xsrf_token"):
                CONFIG["xsrf_token"] = data.get("xsrf_token")
        else:
            cookie_str = content
            pairs = dict(p.split("=", 1) for p in cookie_str.split("; ") if "=" in p)
            sapisid = pairs.get("SAPISID", "")
        _cookie_cache.update({"str": cookie_str, "sapisid": sapisid or None, "mtime": mtime})
        return cookie_str, sapisid if sapisid else None
    except Exception as e:
        log(f"Cookie load error: {e}")
        return _cookie_cache["str"], _cookie_cache["sapisid"]


def save_xsrf_token(token: str):
    """Save recovered xsrf_token into config and cookie file."""
    CONFIG["xsrf_token"] = token
    cookie_file = CONFIG.get("cookie_file")
    if cookie_file and os.path.exists(cookie_file):
        try:
            with open(cookie_file, "r", encoding="utf-8") as f:
                cdata = json.load(f)
            cdata["xsrf_token"] = token
            with open(cookie_file, "w", encoding="utf-8") as f:
                json.dump(cdata, f, indent=2)
        except Exception:
            pass


def fetch_latest_bl() -> str | None:
    """Fetch the latest gemini_bl and SNlM0e (xsrf) from gemini.google.com page."""
    try:
        cookie_str, sapisid = load_cookie()
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        if cookie_str:
            headers["Cookie"] = cookie_str
        if sapisid:
            headers["Authorization"] = make_sapisidhash(sapisid)
        prefix = _account_prefix()
        target_url = f"https://gemini.google.com{prefix}/app"
        req = urllib.request.Request(target_url, headers=headers)
        ctx = _get_ssl_ctx()
        proxy = CONFIG.get("proxy")
        if proxy:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                urllib.request.HTTPSHandler(context=ctx))
            resp = opener.open(req, timeout=15)
        else:
            resp = urllib.request.urlopen(req, context=ctx, timeout=15)
        html = resp.read().decode("utf-8", errors="replace")

        if not CONFIG.get("xsrf_token"):
            m_xsrf = re.search(r'"FdrFJe"\s*:\s*"(-?\d+)"', html)
            if not m_xsrf:
                m_xsrf = re.search(r'"SNlM0e":"([^"]+)"', html)
            if not m_xsrf:
                m_xsrf = re.search(r'SNlM0e[\'":=\s]*([A-Za-z0-9_\-]{10,})', html)
            if m_xsrf:
                save_xsrf_token(m_xsrf.group(1))
                log(f"XSRF token auto-detected from page: {CONFIG['xsrf_token']}")

        m = re.search(r'(boq_assistant-bard-web-server_\d+\.\d+_p\d+)', html)
        if m:
            return m.group(1)
    except Exception as e:
        log(f"BL auto-update fetch failed: {e}")
    return None


def update_bl_if_needed() -> bool:
    """Attempt to fetch and update gemini_bl. Returns True if updated."""
    new_bl = fetch_latest_bl()
    if new_bl and new_bl != CONFIG["gemini_bl"]:
        log(f"BL auto-updated: {CONFIG['gemini_bl']} -> {new_bl}")
        CONFIG["gemini_bl"] = new_bl
        return True
    return False


def make_sapisidhash(sapisid: str) -> str:
    ts = int(time.time())
    h = hashlib.sha1(f"{ts} {sapisid} https://gemini.google.com".encode()).hexdigest()
    return f"SAPISIDHASH {ts}_{h}"


def _account_prefix() -> str:
    """Return the Gemini account path prefix for non-default Google accounts."""
    auth_user = CONFIG.get("auth_user")
    if auth_user is None or auth_user == "":
        return ""
    return f"/u/{auth_user}"


MODEL_IDS = {
    "gemini-3.8-flash": "56fdd199312815e2",
    "gemini-3.7-flash": "56fdd199312815e2",
    "gemini-3.6-flash": "56fdd199312815e2",
    "gemini-3.5-flash": "56fdd199312815e2",
    "gemini-3.1-pro": "e6fa609c3fa255c0",
    "gemini-flash-lite": "8c46e95b1a07cecc",
    "gemini-3.5-flash-thinking": "56fdd199312815e2",
    "gemini-3.5-flash-thinking-lite": "56fdd199312815e2",
    "gemini-auto": None,
}


def build_model_header(model_name: str, model_id: int):
    mid = MODEL_IDS.get(model_name)
    if not mid:
        return None
    try:
        return json.dumps(
            [1, None, None, None, mid, None, None, 0,
             [4, 5, 6, 8, 4, 5, 6, 8], None, None, 2,
             None, None, model_id, 0, str(uuid.uuid4())],
            separators=(",", ":"))
    except Exception:
        return None


def _build_headers(model_name: str = None, model_id: int = None) -> dict:
    account_prefix = _account_prefix()
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://gemini.google.com",
        "Referer": f"https://gemini.google.com{account_prefix}/app",
        "X-Same-Domain": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if account_prefix:
        headers["X-Goog-AuthUser"] = str(CONFIG["auth_user"])
    cookie_str, sapisid = load_cookie()
    if cookie_str:
        headers["Cookie"] = cookie_str
    if sapisid:
        headers["Authorization"] = make_sapisidhash(sapisid)
    if model_name and model_id is not None:
        hdr = build_model_header(model_name, model_id)
        if hdr:
            headers["x-goog-ext-525001261-jspb"] = hdr
    return headers


def _apply_chat_persistence_flags(inner: list) -> None:
    """Apply Gemini Web persistence flags to an outgoing request payload."""
    if CONFIG.get("temporary_chats", False):
        # Match Gemini Web temporary-chat requests.
        inner[41] = [1]
        inner[45] = 1
    else:
        inner[41] = [2]


def _build_payload(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None) -> str:
    inner = [None] * 102
    if file_refs:
        refs = [[None, None, ref] for ref in file_refs]
        inner[0] = [prompt, 0, None, refs, None, None, 0]
    else:
        inner[0] = [prompt, 0, None, None, None, None, 0]
    inner[1] = ["en"]
    inner[2] = ["", "", "", None, None, None, None, None, None, ""]
    inner[6] = [0]
    inner[7] = 1
    inner[10] = 1
    inner[11] = 0
    inner[17] = [[think_mode]]
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    _apply_chat_persistence_flags(inner)
    inner[53] = 0
    inner[59] = str(uuid.uuid4())
    inner[61] = []
    inner[68] = 1
    inner[79] = model_id
    if extra_fields:
        for k, v in extra_fields.items():
            inner[k] = v
    outer = [None, json.dumps(inner)]
    params = {"f.req": json.dumps(outer)}
    if CONFIG.get("xsrf_token"):
        params["at"] = CONFIG["xsrf_token"]
    return urllib.parse.urlencode(params)


def _get_url() -> str:
    reqid = int(time.time()) % 1000000
    account_prefix = _account_prefix()
    return (
        f"https://gemini.google.com{account_prefix}/_/BardChatUi/data/"
        "assistant.lamda.BardFrontendService/StreamGenerate"
        f"?bl={CONFIG['gemini_bl']}&hl=en&_reqid={reqid}&rt=c"
    )


def clean_text(text: str, strip: bool = True) -> str:
    text = re.sub(
        r'```(?:python|javascript|text)\?code_(?:reference|stdout)&code_event_index=\d+\n.*?```\n?',
        '', text, flags=re.DOTALL
    )
    text = re.sub(r'http://googleusercontent\.com/card_content/\d+\n?', '', text)
    return text.strip() if strip else text


def _extract_texts_from_line(line: str) -> list:
    """Parse a single wrb.fr line and return list of text strings found."""
    if '"wrb.fr"' not in line or len(line) < 200:
        return []
    try:
        arr = json.loads(line)
        inner_str = arr[0][2]
        if not inner_str or len(inner_str) < 50:
            return []
        inner = json.loads(inner_str)
        if not (isinstance(inner, list) and len(inner) > 4 and inner[4]):
            return []
        texts = []
        for part in inner[4]:
            if isinstance(part, list) and len(part) > 1 and part[1] and isinstance(part[1], list):
                for t in part[1]:
                    if isinstance(t, str) and t:
                        texts.append(t)
        return texts
    except (json.JSONDecodeError, IndexError, TypeError):
        return []


def extract_response_text(raw: str) -> str:
    """Parse full response to get final text."""
    bard_err = re.search(r'BardErrorInfo\s*\[(\d+)\]', raw)
    if bard_err:
        raise RuntimeError(f"Gemini upstream rejected request: BardErrorInfo [{bard_err.group(1)}]")
    last_text = ""
    for line in raw.split("\n"):
        for t in _extract_texts_from_line(line):
            if len(t) > len(last_text):
                last_text = t
    return clean_text(last_text)


def generate(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None, model_name: str = None) -> str:
    """Non-streaming generation with retry."""
    headers = _build_headers(model_name, model_id)
    body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields).encode()
    url = _get_url()
    ctx = _get_ssl_ctx()
    proxy = CONFIG.get("proxy")

    last_err = None
    for attempt in range(CONFIG["retry_attempts"]):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            if proxy:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                    urllib.request.HTTPSHandler(context=ctx)
                )
                resp = opener.open(req, timeout=CONFIG["request_timeout_sec"])
            else:
                resp = urllib.request.urlopen(req, context=ctx, timeout=CONFIG["request_timeout_sec"])
            raw = resp.read().decode("utf-8", errors="replace")
            return extract_response_text(raw)
        except urllib.error.HTTPError as e:
            if e.code == 405 and update_bl_if_needed():
                url = _get_url()
                log("Retrying with updated BL...")
                last_err = e
                continue
            if e.code == 400:
                try:
                    err_body = e.read().decode("utf-8", errors="replace")
                    m = re.search(r'\["xsrf",\s*"([^"]+)"', err_body)
                    if m:
                        new_xsrf = m.group(1)
                        log(f"XSRF auto-recovered from 400: {new_xsrf}")
                        save_xsrf_token(new_xsrf)
                        body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields).encode()
                        last_err = e
                        continue
                except Exception as ex:
                    log(f"XSRF recovery parse error: {ex}")
            last_err = e
            if attempt < CONFIG["retry_attempts"] - 1:
                log(f"Retry {attempt+1}/{CONFIG['retry_attempts']}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
        except Exception as e:
            last_err = e
            if attempt < CONFIG["retry_attempts"] - 1:
                log(f"Retry {attempt+1}/{CONFIG['retry_attempts']}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
    raise last_err


def generate_stream(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None, model_name: str = None):
    """Streaming generation via httpx with retry on connection failure."""
    if not HAS_HTTPX:
        text = generate(prompt, model_id, think_mode, file_refs, extra_fields, model_name)
        if text:
            yield text
        return

    headers = _build_headers(model_name, model_id)
    body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields)
    url = _get_url()
    client = _get_httpx_client()

    emitted_raw_text = ""
    for attempt in range(CONFIG["retry_attempts"]):
        try:
            with client.stream("POST", url, content=body, headers=headers) as resp:
                if resp.status_code == 405:
                    update_bl_if_needed()
                    raise RuntimeError("HTTP 405, BL updated")
                if resp.status_code == 400:
                    err_text = resp.read().decode("utf-8", errors="replace")
                    m = re.search(r'\["xsrf",\s*"([^"]+)"', err_text)
                    if m:
                        new_xsrf = m.group(1)
                        log(f"XSRF auto-recovered from stream 400: {new_xsrf}")
                        save_xsrf_token(new_xsrf)
                    raise RuntimeError("HTTP 400 Bad Request, recovered XSRF")
                resp.raise_for_status()
                buf = ""
                for chunk in resp.iter_text():
                    buf += chunk
                    if "BardErrorInfo" in buf:
                        bard_err = re.search(r'BardErrorInfo\s*\[(\d+)\]', buf)
                        if bard_err:
                            raise RuntimeError(
                                f"Gemini upstream rejected request: BardErrorInfo [{bard_err.group(1)}]"
                            )
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        for t in _extract_texts_from_line(line):
                            if t == emitted_raw_text or emitted_raw_text.startswith(t):
                                continue
                            if not t.startswith(emitted_raw_text):
                                raise RuntimeError("Gemini stream content changed during retry")
                            delta = clean_text(t[len(emitted_raw_text):], strip=False)
                            emitted_raw_text = t
                            if delta:
                                yield delta
            return
        except Exception as e:
            log(f"Stream error ({e}), retrying via non-streaming fallback...")
            text = generate(prompt, model_id, think_mode, file_refs, extra_fields)
            if text:
                yield text
            return
