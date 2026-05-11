"""
Setup script — run once after cloning.
Installs dependencies, trains models, and generates sample PDFs.

Usage:
    python setup.py
"""

import subprocess
import sys
from pathlib import Path


def run(cmd: list, check: bool = True):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result.returncode == 0


def main():
    print("=" * 60)
    print("  Financial Analyzer — Setup")
    print("=" * 60)

    # 1. Install requirements
    print("\n[1/4] Installing Python dependencies...")
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    # 2. Train spend classifier
    print("\n[2/4] Training spend classifier (char TF-IDF + LinearSVC)...")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from classifier.train import train
        metrics = train()
        print(f"  ✓ Classifier trained. CV F1: {metrics['cv_f1_mean']:.3f}")
    except Exception as e:
        print(f"  ✗ Classifier training failed: {e}")

    # 3. Train CRF description parser
    print("\n[3/4] Training CRF description parser...")
    try:
        from parser.crf_parser import CRFDescriptionParser
        CRFDescriptionParser()
        print("  ✓ CRF parser trained and ready.")
    except Exception as e:
        print(f"  ✗ CRF parser training failed: {e}")

    # 4. Generate sample PDFs for testing
    print("\n[4/4] Generating sample bank statement PDFs...")
    try:
        from data.generate_sample_pdf import generate_hdfc_statement, generate_sbi_statement
        data_dir = Path(__file__).parent / "data"
        generate_hdfc_statement(str(data_dir / "hdfc_sample_statement.pdf"), months=2)
        generate_sbi_statement(str(data_dir / "sbi_sample_statement.pdf"), months=2)
        print("  ✓ Sample PDFs generated in data/")
    except ImportError:
        print("  ⚠ reportlab not installed — skipping sample PDF generation.")
        print("    Install with: pip install reportlab")
        print("    Then run: python data/generate_sample_pdf.py")
    except Exception as e:
        print(f"  ✗ Sample PDF generation failed: {e}")

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("  Start the app: streamlit run app/streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
