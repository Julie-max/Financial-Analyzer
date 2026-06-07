"""
Column Header Classifier — Simple Normalized Lookup

Maps column headers to their semantic roles using normalized text matching.
No ML needed here — there are a finite set of column header variations
across banks, and a lookup table is more reliable and debuggable.

Roles: date | description | debit | credit | balance | amount | reference | ignore
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column role lookup table
# Normalized header text → role
# Covers: SBI, HDFC, ICICI, Axis, CUB, Kotak, PNB, Canara, BOI, Yes Bank,
#         IDBI, Federal, IndusInd, UCO, IOB, Bandhan, RBL, AU Small Finance
# ---------------------------------------------------------------------------
HEADER_ROLE_MAP: Dict[str, str] = {
    # ── DATE ──────────────────────────────────────────────────────────────
    "date": "date",
    "txn date": "date",
    "transaction date": "date",
    "value date": "date",
    "post date": "date",
    "posting date": "date",
    "trans date": "date",
    "tran date": "date",
    "entry date": "date",
    "book date": "date",
    "effective date": "date",
    "process date": "date",
    "instrument date": "date",
    "cheque date": "date",
    "transaction dt": "date",
    "txn dt": "date",
    "val date": "date",
    "value dt": "date",
    "date of transaction": "date",

    # ── DESCRIPTION ───────────────────────────────────────────────────────
    "description": "description",
    "narration": "description",
    "remarks": "description",
    "particulars": "description",
    "transaction remarks": "description",
    "details": "description",
    "transaction details": "description",
    "transaction narration": "description",
    "transaction description": "description",
    "narrative": "description",
    "memo": "description",
    "transaction particulars": "description",
    "cheque details": "description",
    "transaction info": "description",
    "payment details": "description",
    "beneficiary details": "description",
    "txn remarks": "description",
    "txn description": "description",
    "transaction": "description",
    "transactions": "description",

    # ── DEBIT ─────────────────────────────────────────────────────────────
    "debit": "debit",
    "withdrawal": "debit",
    "dr": "debit",
    "withdrawals": "debit",
    "debit amount": "debit",
    "withdrawal amount": "debit",
    "dr amount": "debit",
    "debit inr": "debit",
    "withdrawal inr": "debit",
    "debit rs": "debit",
    "withdrawal rs": "debit",
    "money out": "debit",
    "outflow": "debit",
    "paid out": "debit",
    "debit amt": "debit",
    "wdl": "debit",
    "debit amount inr": "debit",
    "withdrawal amount inr": "debit",
    "debit amount rs": "debit",

    # ── CREDIT ────────────────────────────────────────────────────────────
    "credit": "credit",
    "deposit": "credit",
    "cr": "credit",
    "deposits": "credit",
    "credit amount": "credit",
    "deposit amount": "credit",
    "cr amount": "credit",
    "credit inr": "credit",
    "deposit inr": "credit",
    "credit rs": "credit",
    "deposit rs": "credit",
    "money in": "credit",
    "inflow": "credit",
    "paid in": "credit",
    "credit amt": "credit",
    "dep": "credit",
    "credit amount inr": "credit",
    "deposit amount inr": "credit",
    "credit amount rs": "credit",

    # ── BALANCE ───────────────────────────────────────────────────────────
    "balance": "balance",
    "closing balance": "balance",
    "running balance": "balance",
    "available balance": "balance",
    "bal": "balance",
    "closing bal": "balance",
    "net balance": "balance",
    "ledger balance": "balance",
    "balance inr": "balance",
    "balance rs": "balance",
    "closing balance inr": "balance",
    "running bal": "balance",
    "book balance": "balance",
    "ledger bal": "balance",
    "available bal": "balance",

    # ── AMOUNT (single combined column) ───────────────────────────────────
    "amount": "amount",
    "amount inr": "amount",
    "amount rs": "amount",
    "transaction amount": "amount",
    "txn amount": "amount",
    "amount dr cr": "amount",
    "dr cr amount": "amount",
    "debit credit": "amount",
    "dr cr": "amount",
    "amount dr cr": "amount",
    "net amount": "amount",
    "amount debit credit": "amount",

    # ── REFERENCE ─────────────────────────────────────────────────────────
    "cheque no": "reference",
    "cheque number": "reference",
    "chq no": "reference",
    "chq ref": "reference",
    "ref no": "reference",
    "reference": "reference",
    "reference no": "reference",
    "transaction id": "reference",
    "txn id": "reference",
    "transaction no": "reference",
    "instrument no": "reference",
    "utr no": "reference",
    "utr number": "reference",
    "ref no cheque no": "reference",
    "cheque ref no": "reference",
    "trans id": "reference",
    "ref number": "reference",
    "instrument number": "reference",

    # ── IGNORE (serial numbers, page numbers, etc.) ───────────────────────
    "s no": "ignore",
    "sr no": "ignore",
    "serial no": "ignore",
    "sl no": "ignore",
    "no": "ignore",
    "s n": "ignore",
    "sr": "ignore",
    "page": "ignore",
    "type": "ignore",
    "mode": "ignore",
    "channel": "ignore",
    "branch": "ignore",
    "branch code": "ignore",
}


def _normalize_header(text: str) -> str:
    """Normalize a column header for lookup."""
    if not text:
        return ""
    # Lowercase, strip
    text = text.lower().strip()
    # Remove currency symbols, parentheses, dots, special chars
    text = re.sub(r'[₹$€().,;:\-/\\#*]+', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class ColumnClassifier:
    """
    Column header classifier using normalized lookup.
    Predicts the semantic role of a column from its header text.
    """

    def __init__(self):
        self._lookup = HEADER_ROLE_MAP

    def predict(self, header: str) -> str:
        """Predict the role of a column from its header text."""
        if not header or not header.strip():
            return "ignore"
        normalized = _normalize_header(header)
        if not normalized:
            return "ignore"

        # Exact match
        if normalized in self._lookup:
            return self._lookup[normalized]

        # Substring match — check if any key is contained in the normalized header
        # Sort by length descending so longer (more specific) matches win
        for key in sorted(self._lookup.keys(), key=len, reverse=True):
            if key in normalized:
                return self._lookup[key]

        return "ignore"

    def predict_batch(self, headers: List[str]) -> List[str]:
        """Predict roles for a list of column headers."""
        return [self.predict(h) for h in headers]

    def detect_column_map(self, header_row: List) -> Dict[str, int]:
        """
        Given a header row, return a dict mapping role → column index.
        """
        column_map = {}
        for idx, cell in enumerate(header_row):
            if cell is None:
                continue
            role = self.predict(str(cell))
            if role != "ignore" and role not in column_map:
                column_map[role] = idx

        # If no explicit debit/credit but has 'amount', keep amount
        # If has debit but no credit (or vice versa), treat as single amount
        if "debit" in column_map and "credit" not in column_map:
            column_map["amount"] = column_map.pop("debit")
        elif "credit" in column_map and "debit" not in column_map:
            column_map["amount"] = column_map.pop("credit")

        return column_map


# Module-level singleton
_classifier_instance = None


def get_column_classifier() -> ColumnClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ColumnClassifier()
    return _classifier_instance
