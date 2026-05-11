"""
PDF Text & Table Extractor
Extracts raw tables from bank statement PDFs using pdfplumber.
Falls back to PyMuPDF text extraction if table extraction yields nothing.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pdfplumber
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extracts tables and text from PDF files.
    Primary: pdfplumber extract_table() — spatial coordinate-based table detection.
    Fallback: PyMuPDF raw text extraction.
    """

    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract tables from a PDF file page by page.

        Returns:
            dict with keys:
                - pages_tables: list of raw table data per page (list of list of list)
                - pages_text:   list of raw text per page (fallback)
                - method:       'table' or 'text'
                - page_count:   number of pages
        """
        pdf_path = str(Path(pdf_path).resolve())
        pages_tables = []
        page_count = 0

        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                logger.info(f"pdfplumber: {page_count} pages in {pdf_path}")

                for i, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    if tables:
                        # Take the largest table on the page (the transaction table)
                        largest = max(tables, key=lambda t: len(t))
                        pages_tables.append(largest)
                        logger.debug(f"Page {i+1}: {len(largest)} rows extracted")
                    else:
                        pages_tables.append([])
                        logger.debug(f"Page {i+1}: no table found")

            total_rows = sum(len(t) for t in pages_tables)
            if total_rows > 0:
                logger.info(f"Table extraction: {total_rows} total rows across {page_count} pages")
                return {
                    "pages_tables": pages_tables,
                    "pages_text": [],
                    "method": "table",
                    "page_count": page_count,
                }

        except Exception as e:
            logger.warning(f"pdfplumber table extraction failed: {e}")

        # Fallback: raw text via PyMuPDF
        logger.warning("Falling back to PyMuPDF text extraction")
        pages_text = self._extract_text_pymupdf(pdf_path)
        return {
            "pages_tables": [],
            "pages_text": pages_text,
            "method": "text",
            "page_count": len(pages_text),
        }

    def _extract_text_pymupdf(self, pdf_path: str) -> List[str]:
        """Extract raw text page by page using PyMuPDF."""
        pages_text = []
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda b: (round(b[1] / 10), b[0]))
                lines = [b[4].strip() for b in blocks if b[4].strip()]
                pages_text.append("\n".join(lines))
            doc.close()
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
        return pages_text


def extract_pdf(pdf_path: str) -> Dict[str, Any]:
    """Convenience function."""
    return PDFExtractor().extract(pdf_path)
