"""Gemini StreamGenerate protocol implementation with httpx streaming."""
import json
import time
import uuid
import re
import urllib.request
import urllib.parse
import urllib.error
import ssl
import socket
import os
import hashlib
import tempfile
import threading

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    httpx = None
    HAS_HTTPX = False

from .config import CONFIG

_ssl_ctx = None
_cookie_cache = {"str": "", "sapisid": None, "mtime": 0}
_cookie_lock = threading.Lock()
_refresh_lock = threading.Lock()
_httpx_client = None


def log(msg: str):
    if CONFIG.get("log_requests", True):
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


def _extract_sapisid(cookie_str: str) -> str:
    if not cookie_str:
        return ""
    m = re.search(r"(?:^|;\s*)SAPISID=([^;\s]+)", cookie_str)
    if m:
        return m.group(1)
    m3 = re.search(r"(?:^|;\s*)__Secure-3PAPISID=([^;\s]+)", cookie_str)
    return m3.group(1) if m3 else ""


def _parse_netscape_cookies(raw: str) -> str:
    pairs = {}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("#HttpOnly_"):
            line = line[len("#HttpOnly_"):]
        elif not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 7:
            domain = parts[0].lstrip(".").lower()
            if domain == "google.com" or domain.endswith(".google.com"):
                pairs[parts[5]] = parts[6]
    return "; ".join(f"{k}={v}" for k, v in pairs.items())


def _normalize_model_tickets(raw_tickets) -> dict:
    if not isinstance(raw_tickets, dict):
        return {}
    clean = {}
    for mk, mv in raw_tickets.items():
        if len(clean) >= 32:
            break
        if (
            isinstance(mk, str)
            and isinstance(mv, str)
            and re.fullmatch(r"gemini-[a-z0-9._-]{1,48}", mk.strip())
            and re.fullmatch(r"[0-9a-fA-F]{16}", mv.strip())
        ):
            clean[mk.strip()] = mv.strip().lower()
    return clean


def _normalize_metadata(data: dict) -> dict:
    meta = {}
    for k in ("xsrf_token", "gemini_bl"):
        val = data.get(k)
        if isinstance(val, str) and val.strip() and not any(c in val for c in "\r\n\x00"):
            meta[k] = val.strip()
    au = data.get("auth_user")
    if isinstance(au, (int, str)) and not isinstance(au, bool) and str(au).strip().isdigit():
        meta["auth_user"] = str(au).strip()
    tickets = _normalize_model_tickets(data.get("model_tickets"))
    if tickets:
        meta["model_tickets"] = tickets
    return meta


def update_cookie_payload(data: dict) -> tuple:
    """Validate and apply a pushed cookie payload. Returns (ok: bool, err_msg: str, status: int)."""
    if not isinstance(data, dict):
        return False, "invalid JSON object", 400
    raw_cookie = data.get("cookie")
    if not isinstance(raw_cookie, str) or not raw_cookie.strip():
        return False, "missing or invalid cookie field", 400
    cookie_str = raw_cookie.strip()
    if len(cookie_str) > 16384 or any(c in cookie_str for c in "\r\n\x00"):
        return False, "invalid characters or length in cookie field", 400
    sapisid = _extract_sapisid(cookie_str)
    meta = _normalize_metadata(data)

    with _cookie_lock:
        same_cookie = cookie_str == _cookie_cache.get("str") and (sapisid or None) == _cookie_cache.get("sapisid")
        cur_tickets = dict(CONFIG.get("model_tickets") or {})
        new_tickets = meta.get("model_tickets") or {}
        tickets_changed = any(cur_tickets.get(k) != v for k, v in new_tickets.items())
        scalar_changed = any(meta.get(k) != CONFIG.get(k) for k in meta if k != "model_tickets")
        has_new_meta = scalar_changed or tickets_changed
        if same_cookie and not has_new_meta:
            return True, "", 200

        merged_tickets = dict(cur_tickets)
        merged_tickets.update(new_tickets)

        cf = CONFIG.get("cookie_file")
        mtime = _cookie_cache["mtime"]
        if cf:
            tmp_path = None
            try:
                existing = {}
                if os.path.exists(cf):
                    with open(cf, "r", encoding="utf-8-sig") as rf:
                        raw_existing = rf.read().strip()
                    if raw_existing.startswith("{"):
                        loaded = json.loads(raw_existing)
                        if isinstance(loaded, dict):
                            existing = loaded
                existing["cookie"] = cookie_str
                if sapisid:
                    existing["sapisid"] = sapisid
                disk_tickets = _normalize_model_tickets(existing.get("model_tickets"))
                if disk_tickets or merged_tickets:
                    combined = dict(cur_tickets)
                    combined.update(disk_tickets)
                    combined.update(new_tickets)
                    merged_tickets = _normalize_model_tickets(combined)
                    existing["model_tickets"] = merged_tickets
                for k, v in meta.items():
                    if k != "model_tickets":
                        existing[k] = v
                parent_dir = os.path.dirname(os.path.abspath(cf)) or "."
                fd, tmp_path = tempfile.mkstemp(dir=parent_dir, prefix=".cookie-", suffix=".tmp")
                with os.fdopen(fd, "w", encoding="utf-8") as wf:
                    json.dump(existing, wf, ensure_ascii=False, indent=2)
                os.replace(tmp_path, cf)
                tmp_path = None
                mtime = os.path.getmtime(cf)
            except Exception as e:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                log(f"Cookie persist error: {e}")
                return False, "failed to persist cookie file", 500

        for k, v in meta.items():
            if k != "model_tickets":
                CONFIG[k] = v
        if merged_tickets:
            CONFIG["model_tickets"] = merged_tickets
        _cookie_cache.update({"str": cookie_str, "sapisid": sapisid or None, "mtime": mtime})
        log("Updated cookies via /v1/cookies")
        return True, "", 200


def load_cookie() -> tuple:
    """Load cookie from file with mtime-based caching."""
    with _cookie_lock:
        cookie_file = CONFIG.get("cookie_file")
        if not cookie_file or not os.path.exists(cookie_file):
            if cookie_file is None and not _cookie_cache["str"]:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                candidates = [
                    os.path.join(base_dir, "cookie.json"),
                    os.path.join(base_dir, "cookie.txt"),
                    "./cookie.json",
                    "cookie.json",
                    os.path.expanduser("~/.config/gemini-web2api/cookie.json"),
                ]
                for c in candidates:
                    if c and os.path.exists(c):
                        cookie_file = c
                        CONFIG["cookie_file"] = c
                        break
            if not cookie_file or not os.path.exists(cookie_file):
                return _cookie_cache["str"], _cookie_cache["sapisid"]
        try:
            mtime = os.path.getmtime(cookie_file)
            if mtime == _cookie_cache["mtime"] and _cookie_cache["str"]:
                return _cookie_cache["str"], _cookie_cache["sapisid"]
            with open(cookie_file, "r", encoding="utf-8-sig") as f:
                content = f.read().strip()
            if content.startswith("{"):
                data = json.loads(content)
                cookie_str = data.get("cookie", "")
                sapisid = data.get("sapisid", "") or _extract_sapisid(cookie_str)
                meta = _normalize_metadata(data)
                if meta.get("xsrf_token"):
                    CONFIG["xsrf_token"] = meta["xsrf_token"]
                if meta.get("gemini_bl"):
                    CONFIG["gemini_bl"] = meta["gemini_bl"]
                if meta.get("auth_user") and CONFIG.get("auth_user") in (None, ""):
                    CONFIG["auth_user"] = meta["auth_user"]
                if "model_tickets" in meta:
                    cur_tickets = dict(CONFIG.get("model_tickets") or {})
                    cur_tickets.update(meta["model_tickets"])
                    CONFIG["model_tickets"] = cur_tickets
            elif "\t" in content:
                cookie_str = _parse_netscape_cookies(content)
                sapisid = _extract_sapisid(cookie_str)
            else:
                cookie_str = content
                sapisid = _extract_sapisid(cookie_str)
            _cookie_cache.update({"str": cookie_str, "sapisid": sapisid or None, "mtime": mtime})
            return cookie_str, sapisid if sapisid else None
        except Exception as e:
            log(f"Cookie load error: {e}")
            return _cookie_cache["str"], _cookie_cache["sapisid"]


def save_xsrf_token(token: str, gemini_bl: str = None):
    """Save recovered xsrf_token (and optional gemini_bl) into config and cookie file."""
    if token:
        CONFIG["xsrf_token"] = token
    if gemini_bl:
        CONFIG["gemini_bl"] = gemini_bl
    cookie_file = CONFIG.get("cookie_file")
    if cookie_file and os.path.exists(cookie_file):
        try:
            with open(cookie_file, "r", encoding="utf-8-sig") as f:
                raw = f.read().strip()
            if raw.startswith("{"):
                cdata = json.loads(raw)
                if token:
                    cdata["xsrf_token"] = token
                if gemini_bl:
                    cdata["gemini_bl"] = gemini_bl
                with open(cookie_file, "w", encoding="utf-8") as f:
                    json.dump(cdata, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


_BL_PATTERN = re.compile(
    r"(boq_gemini-web-uiserver_\d+\.\d+_p\d+|boq_assistant-bard-web-server_\d+\.\d+_p\d+)"
)


def _fetch_gemini_page_html(cookie_str: str = "", sapisid: str = None, timeout: int = 15) -> tuple:
    """Fetch gemini.google.com/app and return (html, final_url)."""
    account_prefix = _account_prefix()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if cookie_str:
        headers["Cookie"] = cookie_str
    if sapisid:
        headers["Authorization"] = make_sapisidhash(sapisid)
    req = urllib.request.Request(f"https://gemini.google.com{account_prefix}/app?hl=en", headers=headers)
    ctx = _get_ssl_ctx()
    proxy = CONFIG.get("proxy")
    if proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
            urllib.request.HTTPSHandler(context=ctx),
        )
        resp = opener.open(req, timeout=timeout)
    else:
        resp = urllib.request.urlopen(req, context=ctx, timeout=timeout)
    final_url = resp.geturl() if hasattr(resp, "geturl") else ""
    html = resp.read().decode("utf-8", errors="replace")
    return html, final_url


def _is_login_redirect(final_url: str, html: str) -> bool:
    if final_url and "accounts.google.com" in final_url:
        return True
    if "boq_identityfrontendauthuiserver" in html:
        return True
    return False


def fetch_latest_bl() -> str | None:
    """Fetch the latest gemini_bl and SNlM0e (xsrf) from gemini.google.com page."""
    try:
        cookie_str, sapisid = load_cookie()
        html, final_url = _fetch_gemini_page_html(cookie_str, sapisid, timeout=15)
        if _is_login_redirect(final_url, html):
            log("Gemini /app redirected to accounts.google.com (cookie may lack OSID or session expired); fetching BL anonymously")
            html, _ = _fetch_gemini_page_html("", None, timeout=15)
        else:
            if not CONFIG.get("xsrf_token"):
                m_xsrf = re.search(r'"SNlM0e"\s*:\s*"([^"]+)"', html)
                if not m_xsrf:
                    m_xsrf = re.search(r'"FdrFJe"\s*:\s*"(-?\d+)"', html)
                if m_xsrf:
                    save_xsrf_token(m_xsrf.group(1))
                    log(f"XSRF token auto-detected from page: {CONFIG['xsrf_token']}")

        bl_m = re.search(r'"cfb2h"\s*:\s*"([^"]+)"', html)
        if bl_m and not bl_m.group(1).startswith("boq_identityfrontend"):
            return bl_m.group(1)
        m = _BL_PATTERN.search(html)
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
        save_xsrf_token(None, gemini_bl=new_bl)
        return True
    return False


def refresh_bl_and_xsrf() -> bool:
    """Refresh gemini_bl and xsrf_token from gemini.google.com."""
    old = (CONFIG.get("gemini_bl"), CONFIG.get("xsrf_token"))
    with _refresh_lock:
        if (CONFIG.get("gemini_bl"), CONFIG.get("xsrf_token")) != old:
            return True
        try:
            cookie_str, sapisid = load_cookie()
            html, final_url = _fetch_gemini_page_html(cookie_str, sapisid, timeout=15)
            redirected_to_login = _is_login_redirect(final_url, html)
            if redirected_to_login:
                log("BL/XSRF refresh redirected to accounts.google.com; fetching BL anonymously without overwriting XSRF")
                html, _ = _fetch_gemini_page_html("", None, timeout=15)

            bl_m = re.search(r'"cfb2h"\s*:\s*"([^"]+)"', html)
            if not bl_m or bl_m.group(1).startswith("boq_identityfrontend"):
                bl_m = _BL_PATTERN.search(html)
            if bl_m:
                CONFIG["gemini_bl"] = bl_m.group(1)

            if not redirected_to_login:
                at_m = re.search(r'"SNlM0e"\s*:\s*"([^"]+)"', html)
                if at_m:
                    CONFIG["xsrf_token"] = at_m.group(1)

            changed = (CONFIG.get("gemini_bl"), CONFIG.get("xsrf_token")) != old
            if changed:
                save_xsrf_token(CONFIG.get("xsrf_token"), CONFIG.get("gemini_bl"))
            return changed
        except Exception as e:
            log(f"BL/XSRF auto-refresh failed: {e}")
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


def _build_headers(extra_fields: dict = None) -> dict:
    cookie_str, sapisid = load_cookie()
    account_prefix = _account_prefix()
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://gemini.google.com",
        "Referer": f"https://gemini.google.com{account_prefix}/app",
        "X-Same-Domain": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    }
    if account_prefix:
        headers["X-Goog-AuthUser"] = str(CONFIG["auth_user"])
    if extra_fields and extra_fields.get("_ticket") and CONFIG.get("send_model_ticket", True):
        headers["X-Goog-Ext-525001261-Jspb"] = extra_fields["_ticket"]
        headers["X-Goog-Ext-73010989-Jspb"] = "[0]"
        headers["X-Goog-Ext-73010990-Jspb"] = "[0,0,0]"
    if cookie_str:
        headers["Cookie"] = cookie_str
    if sapisid:
        headers["Authorization"] = make_sapisidhash(sapisid)
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
    if extra_fields and extra_fields.get("_ticket") and CONFIG.get("send_model_ticket", True):
        variant = extra_fields.get("_variant", 1)
        inner[17] = [[0]]
        inner[41] = [1]
        inner[80] = variant
        inner[91] = 0
        inner[96] = 1 if variant != 1 else 0
        if model_id == 8:
            inner[6] = [1]
            inner[30] = []
            inner[49] = 40
            inner[68] = 2
            inner[83] = 1
            inner[91] = 1
            inner[96] = 1
            inner[98] = 1
    if extra_fields:
        for k, v in extra_fields.items():
            if isinstance(k, int):
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


_CANNED_ERROR_RE = re.compile(
    r"^\W*(?:i'?m having a hard time fulfilling your request"
    r"|sorry,? something went wrong"
    r"|something went wrong\.? please try"
    r"|i seem to be encountering an error"
    r"|i encountered an error doing what you asked"
    r"|an error occurred\.? please try"
    r"|i'?m unable to (?:process|complete) (?:that|this|your) request right now)"
    r"[^\n]{0,120}$",
    re.IGNORECASE,
)


class _TicketRejectedError(Exception):
    pass


def _is_canned_error(text: str) -> bool:
    if not text:
        return False
    s = text.strip().replace("\u2019", "'")
    if not s or len(s) > 300:
        return False
    return bool(_CANNED_ERROR_RE.match(s))


def _strip_ticket(extra_fields: dict) -> dict:
    if not extra_fields or not extra_fields.get("_ticket"):
        return extra_fields
    return {k: v for k, v in extra_fields.items() if k not in ("_ticket", "_variant")}


def _ticket_timeout() -> float:
    try:
        req_t = float(CONFIG.get("request_timeout_sec") or 180)
    except (ValueError, TypeError):
        req_t = 180.0
    raw_tt = CONFIG.get("ticket_timeout_sec", 25)
    if raw_tt is None:
        return max(1.0, min(req_t, 25.0))
    try:
        tt = float(raw_tt)
    except (ValueError, TypeError):
        tt = 25.0
    if tt <= 0:
        return max(1.0, req_t)
    return max(1.0, min(req_t, tt))


def _is_timeout_error(err: BaseException) -> bool:
    if isinstance(err, urllib.error.HTTPError):
        return False
    if HAS_HTTPX and httpx is not None:
        if isinstance(err, (httpx.HTTPStatusError, httpx.ConnectTimeout, httpx.PoolTimeout)):
            return False
    if isinstance(err, (TimeoutError, socket.timeout)):
        return True
    if isinstance(err, urllib.error.URLError) and isinstance(getattr(err, "reason", None), (TimeoutError, socket.timeout)):
        return True
    if HAS_HTTPX and httpx is not None and isinstance(err, httpx.TimeoutException):
        return True
    return "timed out" in str(err).lower() or "timeout" in type(err).__name__.lower()


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


def _ensure_extra_ticket(extra_fields: dict = None, model_name: str = None) -> dict:
    if model_name and (not extra_fields or "_ticket" not in extra_fields):
        from .models import resolve_model
        _, _, _, _, resolved_extra = resolve_model(model_name)
        if resolved_extra:
            merged = dict(resolved_extra)
            if extra_fields:
                merged.update(extra_fields)
            return merged
    return extra_fields


def generate(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None, model_name: str = None) -> str:
    """Non-streaming generation with retry."""
    load_cookie()
    extra_fields = _ensure_extra_ticket(extra_fields, model_name)
    ctx = _get_ssl_ctx()
    proxy = CONFIG.get("proxy")
    refreshed = False

    last_err = None
    attempts = max(1, int(CONFIG.get("retry_attempts", 3)))
    attempt = 0
    while attempt < attempts:
        body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields).encode()
        url = _get_url()
        headers = _build_headers(extra_fields)
        has_ticket = bool(extra_fields and extra_fields.get("_ticket") and CONFIG.get("send_model_ticket", True))
        req_timeout = _ticket_timeout() if has_ticket else CONFIG["request_timeout_sec"]
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            if proxy:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                    urllib.request.HTTPSHandler(context=ctx)
                )
                resp = opener.open(req, timeout=req_timeout)
            else:
                resp = urllib.request.urlopen(req, context=ctx, timeout=req_timeout)
            raw = resp.read().decode("utf-8", errors="replace")
            text = extract_response_text(raw)
            if has_ticket and _is_canned_error(text):
                extra_fields = _strip_ticket(extra_fields)
                log("Model ticket returned canned error; falling back without X-Goog-Ext-525001261-Jspb header")
                continue
            return text
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 400 and not refreshed:
                try:
                    err_body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
                    m = re.search(r'\["xsrf",\s*"([^"]+)"', err_body)
                    if m:
                        refreshed = True
                        save_xsrf_token(m.group(1))
                        log(f"XSRF auto-recovered from 400: {m.group(1)}")
                        continue
                except Exception:
                    pass
            if e.code in (400, 405) and not refreshed and refresh_bl_and_xsrf():
                refreshed = True
                log(f"HTTP {e.code}: refreshed BL/XSRF, retrying immediately...")
                continue
            attempt += 1
            if attempt < attempts:
                log(f"Retry {attempt}/{attempts}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
        except Exception as e:
            if has_ticket and (_is_timeout_error(e) or (isinstance(e, RuntimeError) and "BardErrorInfo" in str(e))):
                extra_fields = _strip_ticket(extra_fields)
                log(f"Model ticket rejected or timed out ({e}); falling back without X-Goog-Ext-525001261-Jspb header")
                continue
            last_err = e
            attempt += 1
            if attempt < attempts:
                log(f"Retry {attempt}/{attempts}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
    raise last_err


def generate_stream(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None, model_name: str = None):
    """Streaming generation via httpx with retry on connection failure."""
    extra_fields = _ensure_extra_ticket(extra_fields, model_name)
    if not HAS_HTTPX:
        text = generate(prompt, model_id, think_mode, file_refs, extra_fields)
        if text:
            yield text
        return

    load_cookie()
    client = _get_httpx_client()
    refreshed = False

    last_err = None
    emitted_raw_text = ""
    yielded_any = False
    attempts = max(1, int(CONFIG.get("retry_attempts", 3)))
    attempt = 0
    while attempt < attempts:
        body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields)
        url = _get_url()
        headers = _build_headers(extra_fields)
        has_ticket = bool(extra_fields and extra_fields.get("_ticket") and CONFIG.get("send_model_ticket", True))
        req_timeout = _ticket_timeout() if has_ticket else CONFIG["request_timeout_sec"]
        pending = None
        try:
            with client.stream("POST", url, content=body, headers=headers, timeout=req_timeout) as resp:
                if resp.status_code in (400, 405) and not refreshed and not yielded_any:
                    refreshed = True
                    if resp.status_code == 400:
                        try:
                            err_text = resp.read().decode("utf-8", errors="replace")
                            m = re.search(r'\["xsrf",\s*"([^"]+)"', err_text)
                            if m:
                                save_xsrf_token(m.group(1))
                                log(f"XSRF auto-recovered from stream 400: {m.group(1)}")
                                continue
                        except Exception:
                            pass
                    if refresh_bl_and_xsrf():
                        log(f"Stream HTTP {resp.status_code}: refreshed BL/XSRF, retrying immediately...")
                        continue
                resp.raise_for_status()
                buf = ""
                for chunk in resp.iter_text():
                    buf += chunk
                    if "BardErrorInfo" in buf:
                        bard_err = re.search(r'BardErrorInfo\s*\[(\d+)\]', buf)
                        if bard_err:
                            if not yielded_any and has_ticket:
                                raise _TicketRejectedError(f"BardErrorInfo [{bard_err.group(1)}]")
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
                            if has_ticket and not yielded_any:
                                if len(t) <= 300:
                                    pending = t
                                    continue
                                pending = None
                            delta = clean_text(t[len(emitted_raw_text):], strip=False)
                            emitted_raw_text = t
                            if delta:
                                yielded_any = True
                                yield delta
                if pending is not None:
                    if _is_canned_error(clean_text(pending)):
                        raise _TicketRejectedError("canned error")
                    delta = clean_text(pending, strip=False)
                    emitted_raw_text = pending
                    if delta:
                        yielded_any = True
                        yield delta
            return
        except _TicketRejectedError as e:
            extra_fields = _strip_ticket(extra_fields)
            log(f"Stream model ticket rejected ({e}); falling back without X-Goog-Ext-525001261-Jspb header")
            continue
        except Exception as e:
            if not yielded_any and has_ticket and _is_timeout_error(e):
                extra_fields = _strip_ticket(extra_fields)
                log(f"Stream model ticket timed out ({e}); falling back without X-Goog-Ext-525001261-Jspb header")
                continue
            if yielded_any:
                raise
            last_err = e
            attempt += 1
            if attempt < attempts:
                log(f"Stream retry {attempt}/{attempts}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
    raise last_err
