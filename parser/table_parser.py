"""
Rule-Based Table Parser
Converts raw pdfplumber table rows into structured transaction dicts.

No LLM. No neural network.
Uses:
  - Keyword-based column header detection  (handles any bank layout)
  - Schema detection                        (two-column / DR-CR / signed)
  - UPI remark cleaner                      (extracts payee from UPI strings)
  - Date pattern validation                 (filters header/footer rows)
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column keyword maps
# Maps column header keywords → semantic role
# Covers ICICI, HDFC, SBI, Axis, Kotak, Yes Bank, IndusInd, Canara, PNB
# ---------------------------------------------------------------------------
COLUMN_KEYWORDS = {
    "date": [
        "date", "txn date", "transaction date", "value date",
        "posting date", "trans date", "tran date", "post date",
        "entry date", "book date", "effective date", "process date",
    ],
    "description": [
        "description", "narration", "remarks", "particulars",
        "transaction remarks", "details", "transaction details",
        "cheque details", "transaction narration", "memo",
        "transaction description", "narrative", "transaction",
    ],
    "debit": [
        "debit", "withdrawal", "dr", "withdrawals", "debit amount",
        "withdrawal amount", "debit (", "dr amount", "paid out",
        "debit(", "₹ debit", "rs. debit", "wdl", "money out",
        "outflow", "expense", "debit amt",
    ],
    "credit": [
        "credit", "deposit", "cr", "deposits", "credit amount",
        "deposit amount", "credit (", "cr amount", "paid in",
        "credit(", "₹ credit", "rs. credit", "dep", "money in",
        "inflow", "credit amt",
    ],
    "balance": [
        "balance", "closing balance", "running balance",
        "available balance", "bal", "balance (", "closing bal",
        "balance(", "net balance", "ledger balance", "balance( )",
        "balance()",
    ],
    "amount": [
        "amount", "amount( )", "amount()", "amount (", "amount(",
        "transaction amount", "txn amount",
    ],
    "reference": [
        "cheque", "chq", "ref", "reference", "cheque no",
        "chq no", "ref no", "cheque number", "ref no/",
        "cheque no.", "ref no.", "chq/ref", "transaction id",
        "txn id", "transaction no", "instrument no", "trans id",
    ],
    "serial": [
        "s no", "sno", "s.no", "sr no", "serial", "no.",
        "#", "sl no", "sr.", "s/n",
    ],
}

# Date patterns used to validate transaction rows
DATE_PATTERNS = [
    re.compile(r"^\d{2}/\d{2}/\d{4}$"),         # DD/MM/YYYY
    re.compile(r"^\d{2}-\d{2}-\d{4}$"),         # DD-MM-YYYY
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),         # YYYY-MM-DD
    re.compile(r"^\d{2}-[A-Za-z]{3}-\d{4}$"),   # DD-Mon-YYYY
    re.compile(r"^\d{2}\s[A-Za-z]{3}\s\d{4}$"), # DD Mon YYYY
    re.compile(r"^\d{2}/[A-Za-z]{3}/\d{4}$"),   # DD/Mon/YYYY
    re.compile(r"^\d{2}\.\d{2}\.\d{4}$"),       # DD.MM.YYYY
    re.compile(r"^\d{2}/\d{2}/\d{2}$"),         # DD/MM/YY
    re.compile(r"^\d{2}-\d{2}-\d{2}$"),         # DD-MM-YY
    re.compile(r"^\d{4}/\d{2}/\d{2}$"),         # YYYY/MM/DD
    re.compile(r"^[A-Za-z]{3}\s\d{2},\s\d{4}$"), # Mon DD, YYYY
    re.compile(r"^\d{2}\s[A-Za-z]{3}$"),        # DD Mon (no year, some banks)
]

# DR/CR suffix pattern (SBI style: "1,250.00 DR")
DR_CR_PATTERN = re.compile(r"^([\d,]+\.?\d*)\s*(DR|CR)$", re.IGNORECASE)

# Amount cleanup
AMOUNT_CLEAN = re.compile(r"[₹$€,\s]")


class TableParser:
    """
    Parses raw pdfplumber table rows into structured transaction dicts.
    Works across different bank statement layouts automatically.
    """

    def parse_pages(self, pages_tables: List[List[List]]) -> List[Dict[str, Any]]:
        """
        Parse all pages of raw table data into transaction dicts.
        """
        all_transactions = []
        column_map = None

        # First pass: find column map by scanning ALL pages
        if column_map is None:
            for page_idx, table in enumerate(pages_tables):
                if not table:
                    continue
                header_row = self._find_header_row(table)
                if header_row is not None:
                    column_map = self._detect_columns(header_row)
                    logger.info(f"Page {page_idx+1}: column map from header: {column_map}")
                    break

        if column_map is None:
            # No header found anywhere — infer from ALL pages combined
            all_rows = [row for table in pages_tables for row in (table or [])]
            column_map = self._infer_columns_from_data(all_rows)
            if column_map:
                logger.info(f"Column map inferred from all pages: {column_map}")

        if column_map is None:
            logger.warning("Could not detect column map from any page")
            return []

        # Second pass: parse all transaction rows
        for page_idx, table in enumerate(pages_tables):
            if not table:
                continue
            for row in table:
                txn = self._parse_row(row, column_map)
                if txn is not None:
                    all_transactions.append(txn)

        logger.info(f"TableParser: {len(all_transactions)} raw transactions extracted")
        return all_transactions

    # ------------------------------------------------------------------
    # Column detection
    # ------------------------------------------------------------------

    def _find_header_row(self, table: List[List]) -> Optional[List]:
        """Find the header row in a table (first row with column keywords)."""
        for row in table[:5]:
            if row and self._is_header_row(row):
                return row
        return None

    def _is_header_row(self, row: List) -> bool:
        """Check if a row looks like a column header row."""
        if not row:
            return False
        text = " ".join(str(c).lower() for c in row if c)
        keyword_hits = sum(
            1 for keywords in COLUMN_KEYWORDS.values()
            for kw in keywords
            if kw in text
        )
        # Require at least 2 keyword hits
        # Relaxed from 3 to handle banks with fewer standard column names
        # e.g. a table with just "Date | Details | Amount | Balance" = 3 hits
        has_date_keyword = any(kw in text for kw in COLUMN_KEYWORDS["date"])
        has_amount_keyword = any(
            kw in text for kw in
            COLUMN_KEYWORDS["debit"] + COLUMN_KEYWORDS["credit"] + COLUMN_KEYWORDS["balance"] + COLUMN_KEYWORDS.get("amount", [])
        )
        # Accept if: 2+ hits with a date keyword, OR 3+ hits with an amount keyword
        return (keyword_hits >= 2 and has_date_keyword) or (keyword_hits >= 3 and has_amount_keyword)

    def _detect_columns(self, header_row: List) -> Dict[str, int]:
        """
        Map column roles to their index positions from a header row.
        Uses longest-match priority to avoid 'transaction id' matching 'description'.
        """
        column_map = {}
        # Score each cell against each role — longer keyword match wins
        for idx, cell in enumerate(header_row):
            if cell is None:
                continue
            cell_lower = str(cell).lower().strip()
            best_role = None
            best_len = 0
            for role, keywords in COLUMN_KEYWORDS.items():
                if role in column_map:
                    continue
                for kw in keywords:
                    if kw in cell_lower and len(kw) > best_len:
                        best_len = len(kw)
                        best_role = role
            if best_role:
                column_map[best_role] = idx

        # Detect schema: two-column vs single amount column
        if "debit" not in column_map and "credit" not in column_map:
            if "balance" in column_map:
                balance_idx = column_map["balance"]
                if balance_idx > 0:
                    column_map["amount"] = balance_idx - 1

        return column_map

    def _infer_columns_from_data(self, table: List[List]) -> Optional[Dict[str, int]]:
        """
        Infer column positions from actual data rows when no header is found.
        Scans multiple rows to detect both debit and credit columns.
        """
        # Collect all transaction rows first
        txn_rows = [
            row for row in table
            if row and len(row) >= 3
            and any(c and self._is_date(str(c).strip()) for c in row)
        ]
        if not txn_rows:
            return None

        # Use first row to establish structure
        first_row = txn_rows[0]
        n_cols = len(first_row)

        date_cols = [i for i, c in enumerate(first_row) if c and self._is_date(str(c).strip())]
        if not date_cols:
            return None

        date_col = date_cols[0]

        # Description: first non-date, non-numeric column with substantial text
        desc_col = None
        for i, c in enumerate(first_row):
            if i not in date_cols and c and len(str(c)) > 5:
                try:
                    float(str(c).replace(',', '').replace('₹', '').strip())
                except ValueError:
                    desc_col = i
                    break
        if desc_col is None:
            desc_col = date_cols[-1] + 1 if date_cols[-1] + 1 < n_cols else 2

        # Scan ALL rows to find which columns ever have numeric values
        # This catches credit columns that are '-' in debit rows
        col_has_value = {i: False for i in range(n_cols)}
        for row in txn_rows[:20]:  # scan up to 20 rows
            for i, c in enumerate(row):
                if c and i not in date_cols and i != desc_col:
                    val = self._to_float(str(c))
                    if val is not None and val > 0:
                        col_has_value[i] = True

        amount_cols = sorted([i for i, has in col_has_value.items() if has])

        if not amount_cols:
            return None

        # Balance: last amount column
        balance_col = amount_cols[-1]
        pre_balance = [c for c in amount_cols if c < balance_col]

        column_map = {
            "date": date_col,
            "description": desc_col,
            "balance": balance_col,
        }

        if len(pre_balance) >= 2:
            # Two columns before balance = debit + credit
            column_map["debit"] = pre_balance[0]
            column_map["credit"] = pre_balance[1]
        elif len(pre_balance) == 1:
            # Single amount column — sign from WDL/DEP prefix
            column_map["amount"] = pre_balance[0]
        else:
            # Check for dash-placeholder columns (SBI: debit=20.00, credit=-)
            # Look for columns between desc and balance that sometimes have values
            dash_cols = [
                i for i, c in enumerate(first_row)
                if c and str(c).strip() == '-'
                and i > desc_col and i < balance_col
            ]
            if len(dash_cols) >= 2:
                column_map["debit"] = dash_cols[0]
                column_map["credit"] = dash_cols[1]

        logger.info(f"Inferred column map: {column_map}")
        return column_map

    # ------------------------------------------------------------------
    # Row parsing
    # ------------------------------------------------------------------

    def _parse_row(
        self, row: List, column_map: Dict[str, int]
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a single table row into a transaction dict.
        Returns None if the row is not a valid transaction row.
        """
        if not row or len(row) < 2:
            return None

        # Get date — required field
        date_str = self._get_cell(row, column_map, "date")
        if not date_str or not self._is_date(date_str):
            return None

        # Get raw description BEFORE cleaning (needed for sign detection)
        description_raw = self._get_cell(row, column_map, "description") or ""

        # Detect WDL/DEP prefix for sign determination (SBI format)
        raw_upper = description_raw.upper()
        is_withdrawal = raw_upper.startswith(("WDL TFR", "WDL "))
        is_deposit = raw_upper.startswith(("DEP TFR", "DEP ", "CEMTEX DEP"))

        description = self._clean_description(description_raw)

        # Get amount
        amount = self._parse_amount(row, column_map)
        if amount is None:
            return None

        # Apply sign from WDL/DEP prefix when using single amount column
        if "amount" in column_map and "debit" not in column_map:
            if is_withdrawal and amount > 0:
                amount = -amount
            elif is_deposit and amount < 0:
                amount = abs(amount)

        # Get balance
        balance_str = self._get_cell(row, column_map, "balance")
        balance = self._to_float(balance_str)

        return {
            "date": date_str.strip(),
            "description": description,
            "amount": amount,
            "balance": balance,
        }

    def _parse_amount(
        self, row: List, column_map: Dict[str, int]
    ) -> Optional[float]:
        """
        Parse amount handling all four bank schemas:
        1. Two-column: separate debit and credit columns (HDFC, ICICI)
        2. DR/CR suffix: "1,250.00 DR" (SBI passbook)
        3. (Cr)/(Dr) suffix: "10000.00(Cr)" (Axis Bank)
        4. Signed: plain +/- amount
        """
        # Schema 1: separate debit + credit columns
        if "debit" in column_map or "credit" in column_map:
            debit_str = self._get_cell(row, column_map, "debit")
            credit_str = self._get_cell(row, column_map, "credit")

            debit = self._to_float(debit_str)
            credit = self._to_float(credit_str)

            if debit and debit > 0:
                return -debit
            if credit and credit > 0:
                return credit
            return None

        # Schema 2, 3, 4: single amount column
        amount_str = self._get_cell(row, column_map, "amount")
        if not amount_str:
            return None

        amount_str = amount_str.strip()

        # Schema 3: Axis Bank style — "10000.00(Cr)" or "100.00(Dr)"
        axis_match = re.match(r'^([\d,]+\.?\d*)\s*\((Cr|Dr)\)$', amount_str, re.IGNORECASE)
        if axis_match:
            value = self._to_float(axis_match.group(1))
            suffix = axis_match.group(2).upper()
            if value is None:
                return None
            return value if suffix == "CR" else -value

        # Schema 2: SBI style — "1,250.00 DR" or "1,250.00 CR"
        match = DR_CR_PATTERN.match(amount_str)
        if match:
            value = self._to_float(match.group(1))
            suffix = match.group(2).upper()
            if value is None:
                return None
            return -value if suffix == "DR" else value

        # Schema 4: plain signed or unsigned amount
        return self._to_float(amount_str)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_cell(
        self, row: List, column_map: Dict[str, int], role: str
    ) -> Optional[str]:
        """Safely get a cell value by role."""
        idx = column_map.get(role)
        if idx is None or idx >= len(row):
            return None
        val = row[idx]
        if val is None:
            return None
        return str(val).strip()

    def _is_date(self, value: str) -> bool:
        """Check if a string looks like a date."""
        value = value.strip()
        return any(p.match(value) for p in DATE_PATTERNS)

    def _to_float(self, value: Any) -> Optional[float]:
        """Convert a value to float, handling currency formatting."""
        if value is None:
            return None
        s = str(value).strip()
        if not s or s.lower() in ("", "null", "none", "n/a", "-", "0.0"):
            return None
        # Strip Axis Bank (Cr)/(Dr) suffix from balance column
        # e.g. "1011000.00(Cr)" → "1011000.00"
        s = re.sub(r'\s*\((Cr|Dr)\)$', '', s, flags=re.IGNORECASE).strip()
        # Remove currency symbols, commas, spaces
        s = AMOUNT_CLEAN.sub("", s)
        # Handle parenthetical negatives: (1,250.00) → -1250.00
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        try:
            val = float(s)
            return val if val != 0.0 else None
        except ValueError:
            return None

    def _clean_description(self, value: str) -> str:
        """
        Extract merchant name from raw transaction description.
        Uses CRF parser (Conditional Random Fields) as primary method.
        Falls back to minimal rule-based cleaning if CRF unavailable.
        """
        if not value:
            return "UNKNOWN"

        # Collapse newlines and extra whitespace
        value = " ".join(value.split())

        # Use CRF parser — the ML-based sequence labeler
        try:
            from parser.crf_parser import extract_payee
            result = extract_payee(value)
            if result and result != "UNKNOWN" and len(result) > 1:
                return result
        except Exception as e:
            logger.warning(f"CRF parser failed: {e}, falling back to rules")

        # Fallback: minimal rule-based cleaning
        return self._rule_based_clean(value)

    def _rule_based_clean(self, value: str) -> str:
        """Minimal fallback description cleaner used when CRF is unavailable."""
        import re as _re

        # Strip known bank prefixes
        all_prefixes = (
            "WDL TFR ", "DEP TFR ", "CEMTEX DEP ",
            "TO ONL ", "BY ONL ", "TO POS:", "TO ATM WDL:", "TO ATM ",
            "BY ATM ", "BY CREDIT:", "BY NEFT TRF:", "TO NEFT TRF:",
            "BY ", "TO ",
        )
        for prefix in all_prefixes:
            if value.upper().startswith(prefix.upper()):
                value = value[len(prefix):].strip()
                break

        # Strip SBI suffix pattern
        value = _re.sub(r'\s+\d{10,}\s+AT\s+\d+.*$', '', value, flags=_re.IGNORECASE).strip()

        # UPI: extract payee field
        # Handles: UPI/DR/..., UPI/CR/..., UPIAB/..., UPIAR/...
        if value.upper().startswith("UPI"):
            parts = value.split("/")
            if len(parts) >= 2:
                second = parts[1].strip().upper()
                # Standard SBI/CUB: UPI/DR/REFNO/PAYEE or UPI/CR/REFNO/PAYEE
                if second in ("DR", "CR") and len(parts) >= 4:
                    return parts[3].strip().upper() or "UNKNOWN"
                # Axis Bank: UPIAB/REFNO/CR/PAYEE or UPIAR/REFNO/DR/PAYEE
                # parts[0]=UPIAB, parts[1]=REFNO, parts[2]=CR/DR, parts[3]=PAYEE
                if _is_ref_number(second) and len(parts) >= 4:
                    third = parts[2].strip().upper()
                    if third in ("CR", "DR") and len(parts) >= 4:
                        return parts[3].strip().upper() or "UNKNOWN"
                # Standard ICICI: UPI/PAYEE/VPA/...
                if not _is_ref_number(second) and second not in ("DR", "CR"):
                    return second

        # NEFT: find meaningful part
        if value.upper().startswith("NEFT"):
            parts = value.split("-")
            for part in parts[2:]:
                part = part.strip()
                if len(part) > 4 and not part.isdigit():
                    return part.upper()

        # BIL: extract payee
        if value.upper().startswith("BIL/"):
            parts = value.split("/")
            if len(parts) >= 3:
                second = parts[1].strip().upper()
                if second in ("INFT", "NEFT", "IMPS"):
                    for part in parts[3:]:
                        part = part.strip()
                        if part and len(part) > 2:
                            return part.upper()
                else:
                    payee = parts[2].strip()
                    if payee and len(payee) > 3:
                        return payee.upper()

        # VPS/IPS
        if value.upper().startswith(("VPS/", "IPS/")):
            parts = value.split("/")
            if len(parts) >= 2:
                return parts[1].strip().upper()

        # Generic: remove trailing ref numbers
        cleaned = _re.sub(r'\s+[A-Z0-9]{8,}:?$', '', value).strip()
        if cleaned and cleaned != value:
            return cleaned.upper()

        return value.upper().strip()


def _is_ref_number(s: str) -> bool:
    """
    Check if a string is a reference/transaction number (not a merchant name).
    Examples: '528213654253', 'EGZ1180606', 'INDBN52025073104653733'
    """
    s = s.strip()
    if not s:
        return False
    if s.isdigit():
        return True
    # Mostly digits (>60% numeric) and length > 6
    digit_ratio = sum(c.isdigit() for c in s) / len(s)
    if digit_ratio > 0.6 and len(s) > 6:
        return True
    # Known ref number prefixes
    ref_prefixes = ("INDBN", "INDBH", "ICICN", "KKBKN", "YESF", "HDFCH")
    if any(s.upper().startswith(p) for p in ref_prefixes):
        return True
    return False


def _extract_vpa_name(vpa: str) -> str:
    """
    Extract a readable merchant name from a UPI VPA (Virtual Payment Address).
    VPAs look like: makemytrip@hdfc, swiggystores@icici, gpay-11265, paytmqr6x5
    Returns a clean name or empty string if not extractable.

    Examples:
        makemytrip@hdfc    → makemytrip
        swiggystores@icici → swiggystores
        gpay-11265         → (empty — not meaningful)
        paytmqr6x5         → (empty — not meaningful)
        MAKEMYTRIP         → makemytrip
        IRCTCAUTOP         → irctcautop
        REDBUS32 R         → redbus
        ZEPTOONLIN         → zeptoonlin
    """
    if not vpa:
        return ""
    vpa = vpa.strip()

    # Extract part before @ sign
    if "@" in vpa:
        name = vpa.split("@")[0].strip()
    else:
        name = vpa

    # Remove trailing digits and special chars
    import re as _re
    name = _re.sub(r'[\d\-_\.]+$', '', name).strip()

    # Skip if it's a generic payment handle (gpay, paytm, etc.)
    generic = ("gpay", "paytm", "phonepe", "upi", "bhim", "ybl", "okaxis",
               "okhdfcbank", "okicici", "oksbi", "ibl", "axl", "pth", "oks",
               "okb", "okh", "axisb", "hdfcbank", "icici", "sbi", "yesb",
               "utib", "hdfc", "indb", "cnrb", "ioba", "ubin", "kkbk",
               "barb", "fdrl", "idib", "mahb", "cosb", "bkid", "srcb",
               "unba", "mahg", "nspb", "cbin", "ibkl")
    generic_prefixes = ("paytmqr", "gpay-", "mab.", "vyapar", "bharatpe",
                        "playstore", "goog-", "gpayrechar", "gpayutili",
                        "redbus1", "redbus32", "zeptoonlin", "zeptomarke",
                        "swiggystor", "airtel-", "payzomato")
    if name.lower() in generic or len(name) <= 2:
        return ""
    if any(name.lower().startswith(p) for p in generic_prefixes):
        return ""

    # Skip if mostly digits
    if sum(c.isdigit() for c in name) / max(len(name), 1) > 0.5:
        return ""

    # Clean trailing space + single letter (e.g. "REDBUS32 R" → "REDBUS")
    import re as _re2
    name = _re2.sub(r'\s+\w$', '', name).strip()
    # Remove trailing digits
    name = _re2.sub(r'\d+$', '', name).strip()

    return name if len(name) > 2 else ""


def parse_tables(pages_tables: List[List[List]]) -> List[Dict[str, Any]]:
    """Convenience function."""
    return TableParser().parse_pages(pages_tables)
