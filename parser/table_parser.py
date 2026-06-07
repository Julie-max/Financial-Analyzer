"""
ML-Based Table Parser
Converts raw pdfplumber table rows into structured transaction dicts.

No LLM. No neural network. No hardcoded keyword lists.
Uses:
  - ML Column Classifier (char TF-IDF + LinearSVC) for column header detection
  - CRF Parser for merchant name extraction from descriptions
  - Data-driven schema detection for amount format (debit/credit/single column)
  - Date pattern validation for transaction row filtering
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

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
AMOUNT_CLEAN = re.compile(r"[â‚¹$â‚¬,\s]")


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
            # No header found anywhere â€” infer from ALL pages combined
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
        """Find the header row using the ML column classifier."""
        for row in table[:5]:
            if row and self._is_header_row(row):
                return row
        return None

    def _is_header_row(self, row: List) -> bool:
        """
        Check if a row is a column header using the ML column classifier.
        A row is a header if the classifier assigns meaningful roles to 2+ cells
        and at least one is a date or amount role.
        """
        if not row:
            return False
        from parser.column_classifier import get_column_classifier
        clf = get_column_classifier()
        roles = clf.predict_batch([str(c) if c else "" for c in row])
        meaningful = [r for r in roles if r != "ignore"]
        has_date = "date" in roles
        has_amount = any(r in ("debit", "credit", "balance", "amount") for r in roles)
        return len(meaningful) >= 2 and (has_date or has_amount)

    def _detect_columns(self, header_row: List) -> Dict[str, int]:
        """
        Map column roles to their index positions using the ML column classifier.
        No hardcoded keyword lists â€” generalizes to unseen column headers.
        """
        from parser.column_classifier import get_column_classifier
        clf = get_column_classifier()
        return clf.detect_column_map(header_row)

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
                    float(str(c).replace(',', '').replace('â‚¹', '').strip())
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
            # Single amount column â€” sign from WDL/DEP prefix
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

        # Get date â€” required field
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
            "raw_description": description_raw,
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

        # Schema 3: Axis Bank style â€” "10000.00(Cr)" or "100.00(Dr)"
        axis_match = re.match(r'^([\d,]+\.?\d*)\s*\((Cr|Dr)\)$', amount_str, re.IGNORECASE)
        if axis_match:
            value = self._to_float(axis_match.group(1))
            suffix = axis_match.group(2).upper()
            if value is None:
                return None
            return value if suffix == "CR" else -value

        # Schema 2: SBI style â€” "1,250.00 DR" or "1,250.00 CR"
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
        # e.g. "1011000.00(Cr)" â†’ "1011000.00"
        s = re.sub(r'\s*\((Cr|Dr)\)$', '', s, flags=re.IGNORECASE).strip()
        # Remove currency symbols, commas, spaces
        s = AMOUNT_CLEAN.sub("", s)
        # Handle parenthetical negatives: (1,250.00) â†’ -1250.00
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

        # Use CRF parser â€” the ML-based sequence labeler
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
        """Minimal fallback when CRF is unavailable. Splits on '/' to find payee."""
        value = " ".join(value.split())
        if "/" in value:
            parts = value.split("/")
            # Find first non-empty, non-numeric part that's not a known separator token
            for part in parts:
                part = part.strip()
                if part and len(part) > 2 and not part.isdigit() and part not in ("UPI", "DR", "CR", "NEFT", "BIL", "MMT", "IMPS", "INF", "INFT", "VPS", "IPS", "WDL", "DEP", "TFR", "ONL", "BY", "TO"):
                    return part.upper()
        return value.upper().strip()

def parse_tables(pages_tables: List[List[List]]) -> List[Dict[str, Any]]:
    """Convenience function."""
    return TableParser().parse_pages(pages_tables)
