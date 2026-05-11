"""
Post-Processor for LLM-extracted transactions.
Validates, normalizes, deduplicates, and standardizes transaction data.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
from dateutil import parser as dateutil_parser

logger = logging.getLogger(__name__)

# Fields required in every transaction
REQUIRED_FIELDS = {"date", "description", "amount", "balance"}

# Accepted date input formats (tried in order before falling back to dateutil)
DATE_FORMATS = [
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%Y/%m/%d",
]


class PostProcessor:
    """
    Cleans and normalises raw transaction dicts produced by the LLM parser.
    """

    def __init__(self, deduplicate: bool = True):
        self.deduplicate = deduplicate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, raw_transactions: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Full post-processing pipeline.

        Steps:
            1. Filter out invalid / incomplete records
            2. Normalise dates → YYYY-MM-DD
            3. Coerce amounts and balances to float
            4. Clean description strings
            5. Deduplicate (optional)
            6. Sort by date

        Returns:
            Cleaned pandas DataFrame.
        """
        if not raw_transactions:
            logger.warning("No transactions to process.")
            return self._empty_df()

        logger.info(f"Post-processing {len(raw_transactions)} raw transactions")

        cleaned = []
        skipped = 0

        for i, txn in enumerate(raw_transactions):
            result = self._process_one(txn, i)
            if result is not None:
                cleaned.append(result)
            else:
                skipped += 1

        logger.info(f"Valid: {len(cleaned)} | Skipped: {skipped}")

        if not cleaned:
            return self._empty_df()

        df = pd.DataFrame(cleaned)

        if self.deduplicate:
            before = len(df)
            df = self._deduplicate(df)
            logger.info(f"Deduplication: {before} → {len(df)} transactions")

        df = df.sort_values("date").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_one(
        self, txn: Dict[str, Any], idx: int
    ) -> Optional[Dict[str, Any]]:
        """Process a single transaction dict. Returns None if invalid."""
        if not isinstance(txn, dict):
            logger.debug(f"Row {idx}: not a dict, skipping")
            return None

        # --- Date ---
        date_str = txn.get("date")
        parsed_date = self._parse_date(date_str)
        if parsed_date is None:
            logger.debug(f"Row {idx}: unparseable date '{date_str}', skipping")
            return None

        # --- Amount ---
        amount = self._to_float(txn.get("amount"))
        if amount is None:
            logger.debug(f"Row {idx}: invalid amount '{txn.get('amount')}', skipping")
            return None

        # --- Balance ---
        balance = self._to_float(txn.get("balance"))  # None is acceptable

        # --- Description ---
        description = self._clean_description(txn.get("description", ""))
        if not description:
            description = "UNKNOWN"

        return {
            "date": parsed_date,
            "description": description,
            "amount": amount,
            "balance": balance,
        }

    def _parse_date(self, value: Any) -> Optional[str]:
        """Parse a date value into YYYY-MM-DD string."""
        if value is None:
            return None

        value = str(value).strip()
        if not value or value.lower() in ("null", "none", "n/a", "-"):
            return None

        # Try known formats first (faster + more predictable)
        for fmt in DATE_FORMATS:
            try:
                dt = datetime.strptime(value, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # Fallback: dateutil fuzzy parser
        try:
            dt = dateutil_parser.parse(value, dayfirst=True)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    def _to_float(self, value: Any) -> Optional[float]:
        """Convert a value to float, handling common formatting issues."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)

        s = str(value).strip()
        if not s or s.lower() in ("null", "none", "n/a", "-"):
            return None

        # Remove currency symbols and thousands separators
        s = s.replace(",", "").replace("₹", "").replace("$", "").replace("€", "")
        s = s.replace("Rs.", "").replace("Rs", "").strip()

        # Handle CR/DR suffixes (common in Indian bank statements)
        is_credit = s.upper().endswith("CR")
        is_debit = s.upper().endswith("DR")
        s = s.rstrip("CRDRcrdr").strip()

        try:
            val = float(s)
            if is_debit and val > 0:
                val = -val
            elif is_credit and val < 0:
                val = abs(val)
            return val
        except ValueError:
            return None

    def _clean_description(self, value: Any) -> str:
        """Normalise a transaction description string."""
        if value is None:
            return ""
        s = str(value).strip()
        # Collapse multiple whitespace
        s = " ".join(s.split())
        # Remove leading/trailing punctuation
        s = s.strip(".,;:-")
        return s

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact duplicate transactions (same date, description, amount)."""
        return df.drop_duplicates(
            subset=["date", "description", "amount"], keep="first"
        )

    def _empty_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "description", "amount", "balance"])


def post_process(
    raw_transactions: List[Dict[str, Any]], deduplicate: bool = True
) -> pd.DataFrame:
    """Convenience function."""
    processor = PostProcessor(deduplicate=deduplicate)
    return processor.process(raw_transactions)
