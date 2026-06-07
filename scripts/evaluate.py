"""
Evaluation script for the Financial Analyzer pipeline.
Run on any bank statement PDF to see extraction + classification results.

Usage:
    python scripts/evaluate.py data/your_statement.pdf
    python scripts/evaluate.py data/*.pdf
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.WARNING)

from utils.pipeline import FinancialPipeline


def evaluate_pdf(pdf_path: str):
    """Run pipeline and display results."""
    pipeline = FinancialPipeline()
    result = pipeline.process(pdf_path)
    df = result['transactions']

    print(f"\n{'='*70}")
    print(f"PDF: {pdf_path}")
    print(f"{'='*70}")
    print(f"Pages: {result['page_count']} | Method: {result['extraction_method']}")
    print(f"Raw rows: {result['raw_count']} → Final: {result['final_count']}")

    if result['errors']:
        for err in result['errors']:
            print(f"  ERROR: {err}")

    if df.empty:
        print("  No transactions extracted.")
        return

    print(f"\nCategory Distribution:")
    for cat, count in df['category'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {cat:<15} {count:>4} ({pct:>5.1f}%)")

    print(f"\nAll Transactions ({len(df)}):")
    print(f"{'Date':<12} {'Description':<40} {'Amount':>10} {'Category':<15}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['date']:<12} {row['description'][:40]:<40} {row['amount']:>10.2f} {row['category']:<15}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/evaluate.py <pdf_path> [pdf_path2] ...")
        print("\nRunning on all PDFs in data/:")
        import glob
        pdfs = glob.glob("data/*.pdf")
        for pdf in sorted(pdfs):
            evaluate_pdf(pdf)
    else:
        for pdf in sys.argv[1:]:
            evaluate_pdf(pdf)
