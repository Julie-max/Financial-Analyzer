"""
End-to-End Integration Pipeline
PDF → Table Extraction → CRF Description Parsing → Post-Processing → Classification

Fully offline. No LLM. No GPU required.
- pdfplumber extracts the transaction table spatially
- CRF (Conditional Random Fields) labels tokens to extract merchant names
- Character n-gram TF-IDF + LinearSVC classifies spend categories
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from parser.pdf_extractor import PDFExtractor
from parser.table_parser import TableParser
from parser.post_processor import PostProcessor
from classifier.predict import TransactionClassifier

logger = logging.getLogger(__name__)


class FinancialPipeline:
    """
    Orchestrates the full financial transaction processing pipeline.

    Usage:
        pipeline = FinancialPipeline()
        result = pipeline.process("statement.pdf")
        df = result["transactions"]  # classified DataFrame
    """

    def __init__(
        self,
        deduplicate: bool = True,
        model_path: Optional[str] = None,
    ):
        self.extractor = PDFExtractor()
        self.table_parser = TableParser()
        self.post_processor = PostProcessor(deduplicate=deduplicate)
        self.classifier = TransactionClassifier(model_path=model_path)

    def process(self, pdf_path: str) -> Dict[str, Any]:
        """
        Run the full pipeline on a PDF file.

        Returns:
            Dict with:
                - transactions:       classified DataFrame
                - raw_count:          number of raw parsed transactions
                - final_count:        number after post-processing
                - extraction_method:  'table' or 'text'
                - page_count:         number of PDF pages
                - errors:             list of any non-fatal errors
        """
        errors = []
        pdf_path = str(Path(pdf_path).resolve())
        logger.info(f"=== Pipeline starting: {pdf_path} ===")

        # ── Step 1: PDF Table Extraction ─────────────────────────────────
        logger.info("Step 1: Extracting tables from PDF...")
        try:
            extraction = self.extractor.extract(pdf_path)
        except Exception as e:
            msg = f"PDF extraction failed: {e}"
            logger.error(msg)
            return self._error_result(msg)

        logger.info(
            f"  Pages: {extraction['page_count']} | "
            f"Method: {extraction['method']}"
        )

        # ── Step 2: Table Row Parsing ─────────────────────────────────────
        logger.info("Step 2: Parsing table rows...")
        try:
            if extraction["method"] == "table":
                raw_transactions = self.table_parser.parse_pages(
                    extraction["pages_tables"]
                )
            else:
                # Text fallback — limited parsing from raw text
                raw_transactions = self._parse_text_fallback(
                    extraction["pages_text"]
                )
                if not raw_transactions:
                    errors.append(
                        "Could not extract tables from this PDF. "
                        "It may be a scanned/image PDF. "
                        "Try converting to text-based PDF first."
                    )
        except Exception as e:
            msg = f"Table parsing failed: {e}"
            logger.error(msg)
            errors.append(msg)
            raw_transactions = []

        logger.info(f"  Raw transactions parsed: {len(raw_transactions)}")

        # Log intermediate structured output
        if raw_transactions:
            import json
            preview = raw_transactions[:3]
            logger.info(
                f"  [PARSED TRANSACTIONS - first 3]:\n"
                + json.dumps(preview, indent=2, default=str)
                + (f"\n  ... and {len(raw_transactions) - 3} more"
                   if len(raw_transactions) > 3 else "")
            )

        # ── Step 3: Post-Processing ───────────────────────────────────────
        logger.info("Step 3: Post-processing and normalizing...")
        try:
            df = self.post_processor.process(raw_transactions)
        except Exception as e:
            msg = f"Post-processing failed: {e}"
            logger.error(msg)
            errors.append(msg)
            df = pd.DataFrame(columns=["date", "description", "amount", "balance"])

        logger.info(f"  Transactions after post-processing: {len(df)}")

        # ── Step 4: Classification ────────────────────────────────────────
        logger.info("Step 4: Classifying transactions...")
        try:
            df = self.classifier.classify_dataframe(df)
        except Exception as e:
            msg = f"Classification failed: {e}"
            logger.error(msg)
            errors.append(msg)
            if not df.empty:
                df["category"] = "Others"

        logger.info(f"=== Pipeline complete: {len(df)} transactions ===")

        return {
            "transactions": df,
            "raw_count": len(raw_transactions),
            "final_count": len(df),
            "extraction_method": extraction["method"],
            "page_count": extraction["page_count"],
            "errors": errors,
        }

    def _parse_text_fallback(self, pages_text: list) -> list:
        """
        Minimal text-based fallback when table extraction fails.
        Attempts to find transaction-like lines in raw text.
        """
        import re
        transactions = []
        date_re = re.compile(
            r"(\d{2}[/-]\d{2}[/-]\d{4}|\d{2}[/-][A-Za-z]{3}[/-]\d{4})"
        )
        amount_re = re.compile(r"[\d,]+\.\d{2}")

        for page_text in pages_text:
            for line in page_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                date_match = date_re.search(line)
                amounts = amount_re.findall(line)
                if date_match and amounts:
                    transactions.append({
                        "date": date_match.group(1),
                        "description": line,
                        "raw_description": line,
                        "amount": None,
                        "balance": None,
                    })
        return transactions

    def _error_result(self, message: str) -> Dict[str, Any]:
        return {
            "transactions": pd.DataFrame(
                columns=["date", "description", "raw_description", "amount", "balance", "category"]
            ),
            "raw_count": 0,
            "final_count": 0,
            "extraction_method": "none",
            "page_count": 0,
            "errors": [message],
        }
