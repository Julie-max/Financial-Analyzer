"""
Bank Statement Metadata Extractor
===================================
Extracts account metadata from the first page of Indian bank statement PDFs.

Design principles:
  1. Use PyMuPDF word-block extraction (preserves spatial layout better than raw text)
  2. Two passes: labeled extraction first (when labels exist), then pattern-only fallback
  3. No hard-coded per-bank formats — patterns are structural (digits, IFSC format, etc.)
  4. Returns partial results rather than failing when some fields are missing

Outputs JSON:
  {
    "account_holder": "BRIAN JULIAN C",
    "account_number": "XXXXXXXX4521",
    "account_type": "Savings Account",
    "bank_name": "HDFC BANK",
    "bank_code": "HDFC",
    "ifsc_code": "HDFC0001234",
    "branch": "Coimbatore Main Branch",
    "statement_period": {"from": "2025-01-01", "to": "2025-06-30"},
    "customer_id": "12345678",
    "phone": "XXXXXXX890",
    "address": "...",
    "_confidence": {"account_number": "high", "ifsc_code": "high", ...}
  }

Usage:
  python parser/metadata_extractor.py data/your_statement.pdf
  python parser/metadata_extractor.py data/your_statement.pdf --debug
"""

import re
import json
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

import fitz          # PyMuPDF — already in requirements
import pdfplumber    # already in requirements

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# KNOWN BANK CODES — used to identify bank name from first page header
# ─────────────────────────────────────────────────────────────────────────────

BANK_SIGNATURES = {
    # name substring (lowercase) → (display_name, code)
    "hdfc":             ("HDFC Bank",               "HDFC"),
    "state bank":       ("State Bank of India",      "SBIN"),
    "sbi":              ("State Bank of India",      "SBIN"),
    "icici":            ("ICICI Bank",               "ICIC"),
    "axis":             ("Axis Bank",                "UTIB"),
    "kotak":            ("Kotak Mahindra Bank",      "KKBK"),
    "yes bank":         ("Yes Bank",                 "YESB"),
    "indusind":         ("IndusInd Bank",            "INDB"),
    "canara":           ("Canara Bank",              "CNRB"),
    "bank of baroda":   ("Bank of Baroda",           "BARB"),
    "union bank":       ("Union Bank of India",      "UBIN"),
    "punjab national":  ("Punjab National Bank",     "PUNB"),
    "federal":          ("Federal Bank",             "FDRL"),
    "south indian":     ("South Indian Bank",        "SIBL"),
    "karur vysya":      ("Karur Vysya Bank",         "KVBL"),
    "city union":       ("City Union Bank",          "CIUB"),
    "tamilnad":         ("Tamilnad Mercantile Bank", "TMBL"),
    "iob":              ("Indian Overseas Bank",     "IOBA"),
    "indian overseas":  ("Indian Overseas Bank",     "IOBA"),
    "central bank":     ("Central Bank of India",   "CBIN"),
    "idfc":             ("IDFC First Bank",          "IDFB"),
    "rbl":              ("RBL Bank",                 "RATN"),
    "bandhan":          ("Bandhan Bank",             "BDBL"),
    "airtel":           ("Airtel Payments Bank",     "AIRP"),
    "paytm":            ("Paytm Payments Bank",      "PYTM"),
}

# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL PATTERNS — these work regardless of labels
# ─────────────────────────────────────────────────────────────────────────────

# Account number: 8-20 digit string, optionally masked with X
# Handles: "123456789012", "XXXX XXXX 4521", "XXXXXXXX7890"
_ACC_NUM_RE = re.compile(
    r'\b([Xx*]{2,}[\s\-]?[Xx*\d]{2,}[\s\-]?\d{4,}|'   # masked: XXXX1234, XXX XXXX 4521
    r'\d{9,20})\b'                                        # plain: 12+ digit number
)

# IFSC code: 4 alpha + 0 + 6 alphanumeric (RBI standard)
_IFSC_RE = re.compile(r'\b([A-Z]{4}0[A-Z0-9]{6})\b', re.IGNORECASE)

# MICR code: 9 digits
_MICR_RE = re.compile(r'\b(\d{9})\b')

# Indian mobile (masked or full): 10 digits, or XXXXXXnnnn
_PHONE_RE = re.compile(r'\b([Xx*]{5,6}[\s\-]?\d{4,5}|[6-9]\d{9})\b')

# CIF / Customer ID: 8-12 digit number that appears near "cif" or "customer"
_CIF_RE = re.compile(r'\b(\d{8,12})\b')

# Statement period — catches various date range formats
_PERIOD_RE = re.compile(
    r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{1,2}\s+[A-Za-z]{3}[a-z]*\s+\d{2,4})'
    r'\s*(?:to|TO|-|–|through)\s*'
    r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{1,2}\s+[A-Za-z]{3}[a-z]*\s+\d{2,4})',
    re.IGNORECASE
)

# Date formats for period normalization
_DATE_FMTS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y",
    "%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d-%B-%Y",
]


# ─────────────────────────────────────────────────────────────────────────────
# LABEL SYNONYMS
# Maps many bank-specific label variants → a single canonical field name.
# This is why pure regex failed: banks spell labels 10 different ways.
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {
    # account_holder
    "account holder":         "account_holder",
    "account holder name":    "account_holder",
    "customer name":          "account_holder",
    "name":                   "account_holder",
    "account name":           "account_holder",
    "client name":            "account_holder",
    "beneficiary name":       "account_holder",
    "primary holder":         "account_holder",

    # account_number
    "account number":         "account_number",
    "account no":             "account_number",
    "account no.":            "account_number",
    "a/c number":             "account_number",
    "a/c no":                 "account_number",
    "a/c no.":                "account_number",
    "savings account no":     "account_number",

    # account_type
    "account type":           "account_type",
    "type of account":        "account_type",
    "product":                "account_type",
    "product type":           "account_type",
    "scheme":                 "account_type",

    # ifsc_code
    "ifsc":                   "ifsc_code",
    "ifsc code":              "ifsc_code",
    "ifsc/rtgs":              "ifsc_code",
    "rtgs/neft code":         "ifsc_code",
    "bank code":              "ifsc_code",     # some banks use this

    # branch
    "branch":                 "branch",
    "branch name":            "branch",
    "home branch":            "branch",
    "bank branch":            "branch",
    "branch address":         "branch",
    "branch code":            "branch_code",
    # Explicitly block partial matches that should NOT map to branch
    # (branch email id, branch phone — these map to nothing useful)

    # customer_id
    "customer id":            "customer_id",
    "customer no":            "customer_id",
    "customer number":        "customer_id",
    "cif":                    "customer_id",
    "cif no":                 "customer_id",
    "cif number":             "customer_id",
    "client id":              "customer_id",

    # phone
    "mobile":                 "phone",
    "mobile no":              "phone",
    "mobile number":          "phone",
    "phone":                  "phone",
    "phone no":               "phone",
    "registered mobile":      "phone",

    # statement period
    "statement period":       "statement_period",
    "period":                 "statement_period",
    "statement from":         "statement_period",   # "Statement From : 01-04-2026 to 08-05-2026"
    "from date":              "_period_from",
    "to date":                "_period_to",
    "statement to":           "_period_to",
    "opening date":           "_period_from",
    "closing date":           "_period_to",

    # address
    "address":                "address",
    "communication address":  "address",
    "registered address":     "address",
}


# ─────────────────────────────────────────────────────────────────────────────
# TEXT EXTRACTION — page 1 only, two methods
# ─────────────────────────────────────────────────────────────────────────────

def _extract_page1_text_blocks(pdf_path: str) -> list[dict]:
    """
    Extract text blocks from page 1 using PyMuPDF.
    Returns list of {"text": str, "x0": float, "y0": float, "x1": float, "y1": float}.
    Sorted top-to-bottom, left-to-right.
    This gives us spatial information that raw text.strip() loses.
    """
    blocks = []
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        raw = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        doc.close()
        for b in raw:
            text = b[4].strip()
            if not text:
                continue
            blocks.append({"text": text, "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3]})
        # Sort top-to-bottom, then left-to-right
        blocks.sort(key=lambda b: (round(b["y0"] / 5) * 5, b["x0"]))
    except Exception as e:
        logger.error(f"PyMuPDF block extraction failed: {e}")
    return blocks


def _extract_page1_lines(pdf_path: str) -> list[str]:
    """
    Fallback: extract page 1 as plain line-separated text using pdfplumber.
    Less spatial but works on more PDFs.
    """
    lines = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    except Exception as e:
        logger.error(f"pdfplumber text extraction failed: {e}")
    return lines


def _blocks_to_lines(blocks: list[dict]) -> list[str]:
    """
    Convert spatial blocks to a list of logical lines.
    Blocks at the same y-level (within tolerance) are merged left-to-right.
    """
    if not blocks:
        return []

    # Group blocks by approximate y-row (10pt tolerance)
    rows: dict[int, list] = {}
    for b in blocks:
        row_key = round(b["y0"] / 10)
        rows.setdefault(row_key, []).append(b)

    lines = []
    for key in sorted(rows):
        row_blocks = sorted(rows[key], key=lambda b: b["x0"])
        # Join text within the row; use "  " for large x-gaps (likely label : value)
        parts = []
        prev_x1 = None
        for b in row_blocks:
            if prev_x1 is not None and b["x0"] - prev_x1 > 20:
                parts.append("  ")
            for sub_line in b["text"].splitlines():
                sub_line = sub_line.strip()
                if sub_line:
                    parts.append(sub_line)
            prev_x1 = b["x1"]
        line = " ".join(p for p in parts if p.strip())
        if line:
            lines.append(line)
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# LABELED EXTRACTION — Pass 1
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_label(raw: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for label matching."""
    s = raw.lower().strip()
    s = re.sub(r'[:\-–/]+$', '', s)   # strip trailing colon / dash
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _try_label_match(text: str) -> tuple[Optional[str], str]:
    """
    Try to split a text chunk into (label, value) using LABEL_MAP.
    Returns (canonical_field, value_string) or (None, "").

    Handles formats:
      "Account Number: 123456789"       → normal label: value
      ": Account Number 35144633021"    → SBI style (label AFTER colon prefix)
      "Account Number  123456789"       → no colon, just gap
      "Name BRIAN JULIAN C"             → no separator at all
    """
    text = text.strip()

    # SBI / some banks put ": FieldName Value" — leading colon means label is AFTER it
    if text.startswith(":"):
        inner = text[1:].strip()
        # inner is now "Account Number 35144633021" — match longest prefix
        words = inner.split()
        for end in range(min(5, len(words)), 0, -1):
            candidate = " ".join(words[:end])
            key = LABEL_MAP.get(_normalize_label(candidate))
            if key:
                value = " ".join(words[end:]).strip()
                return key, value

    # Try colon separator (handles ALL colons in the string, not just first)
    # Find the LAST colon that separates a known label
    colon_positions = [i for i, c in enumerate(text) if c == ":"]
    for idx in colon_positions:
        raw_label = text[:idx].strip()
        value = text[idx + 1:].strip()
        key = LABEL_MAP.get(_normalize_label(raw_label))
        if key and value:
            return key, value

    # Try double-space separator (label  value)
    parts = re.split(r'\s{2,}', text, maxsplit=1)
    if len(parts) == 2:
        raw_label, value = parts[0].strip(), parts[1].strip()
        key = LABEL_MAP.get(_normalize_label(raw_label))
        if key:
            return key, value

    # Try "label word(s) at start" — label is longest matching prefix
    words = text.split()
    for end in range(min(6, len(words)), 0, -1):
        candidate = " ".join(words[:end])
        key = LABEL_MAP.get(_normalize_label(candidate))
        if key:
            value = " ".join(words[end:]).strip()
            return key, value

    return None, ""


def _split_compound_line(line: str) -> list[str]:
    """
    Some banks pack multiple label:value pairs on one line, e.g.:
    "Uncleared Amount : 0.00 : CIF Number 88625629526"
    "Clear Balance : 107.67CR Branch Phone : 8925898023"
    
    Only split when BOTH sides independently contain a known label.
    Otherwise return the original line as-is (avoids destroying valid single pairs).
    """
    segments = re.split(r'\s+:\s+', line)
    if len(segments) <= 1:
        return [line]

    # Rebuild as label:value pairs where possible
    # Strategy: scan each segment; if it starts with a known label, start a new chunk
    result = []
    current = segments[0]
    for seg in segments[1:]:
        # Check if seg starts with a known label (meaning a new pair begins)
        words = seg.split()
        found_label = False
        for end in range(min(4, len(words)), 0, -1):
            candidate = " ".join(words[:end])
            if candidate.lower().strip() in LABEL_MAP:
                # seg is "LABEL rest" — emit current as complete chunk, start new
                result.append(current)
                current = ": " + seg   # prefix with ":" so _try_label_match handles it
                found_label = True
                break
        if not found_label:
            # seg is a value continuation — append back with ":"
            current = current + " : " + seg
    result.append(current)
    return result


def _labeled_extraction(lines: list[str]) -> dict:
    """
    First pass: scan every line for label:value patterns.
    Returns a partial metadata dict with confidence annotations.
    """
    result: dict = {}
    confidence: dict = {}

    # Multi-line address accumulation
    in_address = False
    address_lines: list[str] = []

    # Expand compound lines before processing
    expanded_lines = []
    for line in lines:
        expanded_lines.extend(_split_compound_line(line))

    for i, line in enumerate(expanded_lines):
        if not line:
            continue

        # Stop address collection on new label
        if in_address:
            key_check, _ = _try_label_match(line)
            if key_check and key_check != "address":
                result["address"] = " | ".join(address_lines).strip()
                confidence["address"] = "medium"
                in_address = False
                address_lines = []
            else:
                # continuation of address
                address_lines.append(line)
                continue

        key, value = _try_label_match(line)
        if not key or not value:
            continue

        value = value.strip()
        if not value:
            continue

        # Post-process value by field type
        if key == "branch":
            # Reject if value looks like another label or email domain
            low_val = value.lower()
            if any(kw in low_val for kw in ["email", "phone", "id", "@", "code", "ifsc"]):
                continue
            result[key] = value.strip()
            confidence[key] = "medium"
            continue

        elif key == "account_number":
            # Keep only the account-number-looking part
            m = _ACC_NUM_RE.search(value)
            if m:
                result[key] = m.group(1).replace(" ", "").replace("-", "")
                confidence[key] = "high"
            else:
                result[key] = value
                confidence[key] = "low"

        elif key == "ifsc_code":
            m = _IFSC_RE.search(value.upper())
            if m:
                result[key] = m.group(1).upper()
                confidence[key] = "high"
            else:
                result[key] = value.upper()
                confidence[key] = "low"

        elif key == "customer_id":
            m = _CIF_RE.search(value)
            if m:
                result[key] = m.group(1)
                confidence[key] = "high"
            else:
                result[key] = value
                confidence[key] = "low"

        elif key == "phone":
            m = _PHONE_RE.search(value)
            if m:
                result[key] = m.group(1).replace(" ", "").replace("-", "")
                confidence[key] = "high"
            else:
                result[key] = value
                confidence[key] = "low"

        elif key == "statement_period":
            m = _PERIOD_RE.search(value)
            if m:
                result[key] = {"from": _normalize_date(m.group(1)),
                                "to":   _normalize_date(m.group(2))}
                confidence[key] = "high"
            else:
                result[key] = value
                confidence[key] = "low"

        elif key in ("_period_from", "_period_to"):
            sp = result.setdefault("statement_period", {})
            sub = "from" if key == "_period_from" else "to"
            sp[sub] = _normalize_date(value)
            confidence["statement_period"] = "high"

        elif key == "address":
            address_lines = [value]
            in_address = True

        elif key == "account_holder":
            # Clean up all-caps name; strip trailing account numbers
            # Also strip if value contains another label (blob PDF issue)
            clean = _clean_name(value)
            # If value contains a colon (means more labels got merged in), take only the part before it
            if ":" in clean:
                clean = clean[:clean.index(":")].strip()
                clean = _clean_name(clean)
            if clean and len(clean) >= 3:
                result[key] = clean
                confidence[key] = "high"

        elif key == "branch_code":
            result[key] = value.strip()
            confidence[key] = "high"

        else:
            result[key] = value
            confidence[key] = "medium"
    # Flush address
    if in_address and address_lines:
        result["address"] = " | ".join(address_lines).strip()
        confidence["address"] = "medium"

    result["_confidence"] = confidence
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN-ONLY EXTRACTION — Pass 2 (fallback for unlabeled fields)
# ─────────────────────────────────────────────────────────────────────────────

def _pattern_extraction(lines: list[str], existing: dict) -> dict:
    """
    Second pass: scan for structural patterns in lines where labels were absent.
    Fills in any fields still missing from Pass 1.
    """
    full_text = "\n".join(lines)
    confidence = existing.get("_confidence", {})

    # IFSC — extremely reliable regex
    if "ifsc_code" not in existing:
        m = _IFSC_RE.search(full_text.upper())
        if m:
            existing["ifsc_code"] = m.group(1).upper()
            confidence["ifsc_code"] = "high"

    # Name from dash-separated pattern: "Account Number XXXXXX - FULL NAME" 
    # or "Transactions List - FULL NAME - XXXXXX"
    if "account_holder" not in existing:
        _DASH_NAME_RE = re.compile(r'-\s+([A-Z][A-Z\s\.]{3,50})\s*(?:-|$)')
        for line in lines[:15]:
            m = _DASH_NAME_RE.search(line)
            if m:
                candidate = m.group(1).strip()
                words = candidate.split()
                if 2 <= len(words) <= 5 and all(len(w) >= 2 for w in words):
                    existing["account_holder"] = candidate
                    confidence["account_holder"] = "inferred"
                    break

    # Account number — take first match that isn't the IFSC code digits
    if "account_number" not in existing:
        for m in _ACC_NUM_RE.finditer(full_text):
            candidate = m.group(1).replace(" ", "").replace("-", "").replace("X", "")
            ifsc = existing.get("ifsc_code", "")
            if candidate and candidate not in ifsc and len(candidate) >= 4:
                existing["account_number"] = m.group(1).replace(" ", "").replace("-", "")
                confidence["account_number"] = "medium"
                break

    # Statement period
    if "statement_period" not in existing:
        m = _PERIOD_RE.search(full_text)
        if m:
            existing["statement_period"] = {
                "from": _normalize_date(m.group(1)),
                "to":   _normalize_date(m.group(2)),
            }
            confidence["statement_period"] = "high"

    # MICR (only if no IFSC found yet — less reliable, can collide with acc numbers)
    if "micr_code" not in existing:
        for m in _MICR_RE.finditer(full_text):
            candidate = m.group(1)
            # Avoid matches that look like phone or account numbers
            if (existing.get("account_number", "") not in candidate and
                    not candidate.startswith(("6", "7", "8", "9"))):
                existing["micr_code"] = candidate
                confidence["micr_code"] = "medium"
                break

    existing["_confidence"] = confidence
    return existing


# ─────────────────────────────────────────────────────────────────────────────
# BANK NAME DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_bank(lines: list[str]) -> tuple[str, str]:
    """
    Detect bank name and code from the first few lines of the document.
    The bank name is usually the first or second line (letterhead).
    Returns (bank_name, bank_code).
    """
    # First scan only the top 8 lines (letterhead). If found, return immediately.
    for line in lines[:8]:
        low = line.lower()
        for signature, (name, code) in BANK_SIGNATURES.items():
            if signature in low:
                return name, code

    # Broader scan for bank name not in header (some PDFs put bank name further down)
    # But exclude lines that look like transaction data (contain dates or amounts)
    _TXN_LINE_RE = re.compile(r'\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}|UPI/|NEFT|IMPS|RTGS')
    for line in lines[8:30]:
        if _TXN_LINE_RE.search(line):
            break   # Stop at first transaction line — bank name won't be below here
        low = line.lower()
        for signature, (name, code) in BANK_SIGNATURES.items():
            if signature in low:
                return name, code

    # Last resort: IFSC prefix can tell us the bank
    return "", ""


# ─────────────────────────────────────────────────────────────────────────────
# UNLABELED NAME HEURISTIC
# Some banks print the name without any label — it's just a bold line near the top.
# This is the hardest case. We use position + form to guess.
# ─────────────────────────────────────────────────────────────────────────────

# Looks like an all-caps proper name: 2–5 words, all alpha, min 3 chars each
_NAME_HEURISTIC_RE = re.compile(
    r'^([A-Z][A-Z\s\.]{4,60})$'   # All caps, letters + spaces + dots only, 4–60 chars
)

_IGNORE_AS_NAME = {
    "account statement", "bank statement", "passbook", "statement of account",
    "transaction details", "dear customer", "to whom it may concern",
    "consolidated account statement", "account summary",
}


def _infer_name_from_context(lines: list[str], bank_name: str) -> Optional[str]:
    """
    Heuristic: look for an unlabeled name in the first 25 lines.
    Handles:
    - "Mr. C  BRIANT JULIAN" (SBI style with title prefix)
    - "RAHUL SHARMA" (plain all-caps name)
    Candidates: lines that look like 'FIRSTNAME LASTNAME' or 'Mr./Mrs. NAME'.
    Exclude: bank name itself, common headers, lines with digits.
    """
    bank_words = {w.lower() for w in bank_name.split() if len(w) > 2}

    # Pattern 1: title prefix — "Mr.", "Mrs.", "Ms.", "Dr.", "Shri", "Smt"
    _TITLE_RE = re.compile(
        r'^(Mr\.|Mrs\.|Ms\.|Dr\.|Shri\.?|Smt\.?|Sri\.?)\s+(.+)$', re.IGNORECASE
    )

    for line in lines[:25]:
        line = line.strip()
        if len(line) < 4 or len(line) > 80:
            continue

        # Title prefix match — highest confidence
        m = _TITLE_RE.match(line)
        if m:
            name_part = m.group(2).strip()
            # Strip bank name if it got merged on same line
            # e.g. "Mr. C  BRIANT JULIAN State Bank of India"
            for sig in bank_words:
                name_part = re.sub(rf'\b{re.escape(sig)}\b.*$', '', name_part, flags=re.IGNORECASE).strip()
            name_part = re.sub(r'\s+', ' ', name_part).strip()
            if name_part and len(name_part) >= 4:
                return f"{m.group(1)} {name_part}"

    # Pattern 2: All-caps name without title
    for line in lines[:25]:
        line = line.strip()
        if len(line) < 4 or len(line) > 80:
            continue
        if any(ch.isdigit() for ch in line):
            continue
        if line.lower() in _IGNORE_AS_NAME:
            continue
        line_words = {w.lower() for w in line.split() if len(w) > 2}
        if bank_words and line_words.issubset(bank_words | {"bank", "ltd", "limited"}):
            continue
        m = _NAME_HEURISTIC_RE.match(line)
        if m:
            words = line.split()
            if 2 <= len(words) <= 5 and all(len(w) >= 2 for w in words):
                return line.strip()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_date(date_str: str) -> str:
    """Normalize a date string to YYYY-MM-DD. Returns original if parse fails."""
    from datetime import datetime
    date_str = date_str.strip()
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Try dateutil as last resort
    try:
        from dateutil import parser as du
        return du.parse(date_str, dayfirst=True).strftime("%Y-%m-%d")
    except Exception:
        return date_str


def _clean_name(raw: str) -> str:
    """
    Clean a raw name value:
    - Strip trailing account numbers, dates, or junk
    - Strip trailing known label words that got merged in (single-blob PDFs)
    - Normalize whitespace
    """
    # Remove anything that looks like an account number after the name
    raw = re.sub(r'\s*[\d]{6,}.*$', '', raw)
    # Remove trailing known label words: "Account Number", "Account Type", etc.
    _TRAILING_LABELS = re.compile(
        r'\s+(Account\s+Number|Account\s+Type|Account\s+No|A/C|IFSC|Branch|CIF|'
        r'Customer|Mobile|Phone|Address|Statement|Period|Product|Scheme).*$',
        re.IGNORECASE
    )
    raw = _TRAILING_LABELS.sub('', raw)
    # Remove trailing punctuation
    raw = re.sub(r'[,;\.]+$', '', raw)
    # Normalize whitespace
    raw = re.sub(r'\s+', ' ', raw).strip()
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTOR — PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def extract_metadata(pdf_path: str, debug: bool = False) -> dict:
    """
    Extract account metadata from the first page of a bank statement PDF.

    Args:
        pdf_path: Path to the PDF file.
        debug:    If True, include raw lines in the output under "_debug".

    Returns:
        dict with fields:
          account_holder, account_number, account_type, bank_name, bank_code,
          ifsc_code, branch, statement_period, customer_id, phone, address,
          micr_code, _confidence
        Any field not found will be absent (not null).
    """
    pdf_path = str(Path(pdf_path).resolve())

    # --- Step 1: Get text blocks from page 1 ---
    blocks = _extract_page1_text_blocks(pdf_path)
    lines = _blocks_to_lines(blocks)

    if not lines:
        logger.warning("PyMuPDF returned nothing, falling back to pdfplumber")
        lines = _extract_page1_lines(pdf_path)

    if not lines:
        logger.error("Could not extract any text from page 1")
        return {"_error": "Could not extract text from PDF page 1", "_confidence": {}}

    if debug:
        logger.debug(f"Extracted {len(lines)} lines from page 1")
        for i, line in enumerate(lines[:30]):
            logger.debug(f"  [{i:02d}] {line!r}")

    # --- Step 2: Detect bank ---
    bank_name, bank_code = _detect_bank(lines)

    # --- Step 3: Labeled extraction (Pass 1) ---
    metadata = _labeled_extraction(lines)

    # --- Step 4: Pattern-only fallback (Pass 2) ---
    metadata = _pattern_extraction(lines, metadata)

    # --- Step 5: Bank info ---
    if bank_name:
        metadata["bank_name"] = bank_name
        metadata["bank_code"] = bank_code
        metadata["_confidence"]["bank_name"] = "high"
        metadata["_confidence"]["bank_code"] = "high"
    elif "ifsc_code" in metadata:
        # Derive bank from IFSC prefix (first 4 chars)
        ifsc_prefix = metadata["ifsc_code"][:4].upper()
        for _, (name, code) in BANK_SIGNATURES.items():
            if code == ifsc_prefix:
                metadata["bank_name"] = name
                metadata["bank_code"] = code
                metadata["_confidence"]["bank_name"] = "medium"
                metadata["_confidence"]["bank_code"] = "medium"
                break

    # --- Step 6: Name inference if still missing ---
    if "account_holder" not in metadata and bank_name:
        inferred = _infer_name_from_context(lines, bank_name)
        if inferred:
            metadata["account_holder"] = inferred
            metadata["_confidence"]["account_holder"] = "inferred"

    # --- Step 7: Clean up internal keys ---
    metadata.pop("_period_from", None)
    metadata.pop("_period_to", None)

    # --- Step 8: Debug output ---
    if debug:
        metadata["_debug_lines"] = lines[:40]

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Extract account metadata from a bank statement PDF (page 1)."
    )
    p.add_argument("pdf_path", help="Path to the bank statement PDF")
    p.add_argument(
        "--debug", action="store_true",
        help="Show raw lines extracted from page 1 (useful for troubleshooting)"
    )
    p.add_argument(
        "--out", default=None,
        help="Write JSON output to this file (default: print to stdout)"
    )
    p.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity"
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    pdf = args.pdf_path
    if not Path(pdf).exists():
        print(f"ERROR: File not found: {pdf}", file=sys.stderr)
        sys.exit(1)

    result = extract_metadata(pdf, debug=args.debug)

    # Pretty-print, excluding internal debug lines unless --debug
    output = {k: v for k, v in result.items()
              if not k.startswith("_debug") or args.debug}

    json_str = json.dumps(output, indent=2, ensure_ascii=False)

    if args.out:
        Path(args.out).write_text(json_str, encoding="utf-8")
        print(f"Written to {args.out}")
    else:
        print(json_str)
