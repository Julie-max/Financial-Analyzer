# 💰 Financial Transaction Analyzer

A fully offline bank statement analysis system. Upload any bank statement PDF — the system extracts transactions, classifies spending, and visualizes your finances.

**No LLM. No neural network. No internet required.**

## Architecture

```
Bank Statement PDF
        │
        ▼
pdfplumber — spatial table extraction
        │  raw table rows (with full description preserved)
        ▼
Column Classifier (lookup table) — maps headers to roles
        │
        ▼
CRF Parser — sequence labeling (token → PAYEE / REF_NUM / BANK / VPA / ...)
        │  clean merchant names + raw description preserved
        ▼
Post-Processor — date normalization, deduplication, validation
        │  clean DataFrame with raw_description column
        ▼
VPA Pattern Classifier — extracts VPA from raw, classifies as personal/merchant/qr
        │  appends VPA_PERSONAL / VPA_MERCHANT / VPA_QR signal
        ▼
Two-Stage Classifier:
  Stage 1: Merchant lookup (~150 known merchants → instant category)
  Stage 2: Char n-gram TF-IDF + LinearSVC (everything else, with VPA signal)
        │  categorized DataFrame
        ▼
Streamlit UI — charts, insights, CSV export
```

## ML Components (3 layers, down from 5)

| Layer | Algorithm | Purpose |
|-------|-----------|---------|
| CRF Parser | Conditional Random Fields (sklearn-crfsuite) | Extract merchant name from structured bank descriptions |
| VPA Pattern Detection | Deterministic regex rules | Classify VPA as personal/merchant/QR — strongest P2P signal |
| Spend Classifier | Char n-gram TF-IDF + LinearSVC | Classify into 6 spending categories |

### Why these choices

**CRF for parsing:** Bank descriptions are structured sequences (`UPI/DR/REF/PAYEE/BANK/VPA`). CRF labels tokens in context — generalizes across bank formats without bank-specific rules.

**Deterministic VPA classification:** UPI VPAs have distinct patterns:
- Phone numbers (10 digits) → personal (P2P transfer)
- `q` + digits → personal (PhonePe/GPay P2P)
- `paytmqr*`, `vyapar.*` → QR/merchant (small vendor)
- Brand names (`swiggystores`, `indianrail`) → known merchant

This is better as rules than ML because the patterns are structural, not semantic.

**Character n-gram TF-IDF + LinearSVC for classification:** UPI truncates merchant names to 8 chars (`BOOKMYSH` = BookMyShow). Character n-grams (2-5 chars) capture partial overlaps. The `VPA_PERSONAL` signal strongly pushes truncated personal names toward "Others".

### What was removed (previously 5 layers → now 3)

1. **Column Header ML Classifier** → replaced with a simple normalized lookup table. ~100 known column headers across Indian banks don't need ML.
2. **VPA ML Classifier** → replaced with deterministic pattern matching. VPA formats are structural patterns (phone numbers, QR codes), not semantic — rules are more reliable than a small ML model.

## Key Design Decision: Preserving Raw Description

The raw bank description (e.g., `UPI/DR/REF/PAYEE/BANK/VPA/...`) contains VPA info that strongly signals personal vs merchant. The pipeline preserves this through all stages so the classifier can extract VPA signals at classification time.

This solves the #1 problem: **individual names (P2P payments) now correctly classify as "Others"** because the VPA pattern (phone number, q-prefix) identifies them as personal transfers.

## Project Structure

```
financial-analyzer/
├── app/
│   └── streamlit_app.py          # Streamlit UI
├── classifier/
│   ├── merchant_lookup.py        # Stage 1: ~150 known merchants (substring match)
│   ├── predict.py                # Two-stage classifier + VPA extraction
│   └── train.py                  # Char TF-IDF + LinearSVC training (~250 examples)
├── data/
│   ├── generate_sample_pdf.py    # Generate test PDFs (HDFC + SBI)
│   └── *.pdf                     # Bank statement PDFs
├── models/
│   ├── classifier_pipeline.joblib  # Trained TF-IDF + LinearSVC
│   ├── crf_parser.joblib           # Trained CRF model
│   └── categories.json
├── parser/
│   ├── column_classifier.py      # Normalized lookup for column headers
│   ├── crf_parser.py             # CRF sequence labeler for descriptions
│   ├── pdf_extractor.py          # pdfplumber + PyMuPDF fallback
│   ├── post_processor.py         # Date normalization, dedup, validation
│   └── table_parser.py           # Column detection, row parsing, schema detection
├── scripts/
│   └── evaluate.py               # Evaluation script for any PDF
├── utils/
│   ├── insights.py               # Analytics and natural language insights
│   └── pipeline.py               # End-to-end orchestration
├── requirements.txt
├── setup.py
└── README.md
```

## Setup

```bash
# Python 3.10+
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train models (auto-trains on first run, or manually)
python -c "from classifier.train import train; train()"

# Run the app
streamlit run app/streamlit_app.py

# Evaluate on a specific PDF
python scripts/evaluate.py data/your_statement.pdf
```

## Supported Bank Formats

| Bank | Format | Status |
|------|--------|--------|
| ICICI | UPI/PAYEE/VPA/REMARK/BANK/REF | ✅ Tested |
| SBI | WDL TFR UPI/DR/REF/PAYEE/BANK/VPA AT BRANCH | ✅ Tested |
| City Union Bank | UPI/DR/REF/PAYEE/BANK/VPA/U::00116 | ✅ Tested |
| HDFC | Debit/Credit separate columns | ✅ Tested |
| Axis | UPIAB/REF/CR/PAYEE/BANK | ✅ Trained |
| Any digital PDF | CRF + column detection generalizes | Should work |

## Spending Categories

| Category | Examples |
|----------|---------|
| Food | Swiggy, Zomato, restaurants, groceries, cafes, small food vendors (QR) |
| Travel | Uber, IRCTC, flights, fuel, hotels, cabs, metro, auto rides |
| Shopping | Amazon, Flipkart, clothing, electronics |
| Bills | Electricity, mobile, subscriptions (Netflix, Spotify, LinkedIn), EMIs, insurance |
| Entertainment | Movies, gaming, concerts, Zerodha (investments) |
| Others | Salary, P2P transfers, fund transfers, ATM, investments, unknown |

## Known Limitations & Gaps

| Limitation | Impact | Path to Fix |
|-----------|--------|-------------|
| Training data is embedded (~250 examples) | May misclassify unseen merchant patterns | Add feedback loop, retrain on corrections |
| QR payments to individuals are ambiguous | "MOHANRAJ" via paytmqr could be food vendor or person | Need amount-based heuristics or user correction |
| CRF trained on ~30 labeled sequences | May fail on completely new bank formats | Add more labeled examples from new banks |
| Scanned/image PDFs not supported | No OCR layer | Add Tesseract/EasyOCR if needed |
| No configurable categories | Fixed at 6 | Add user-defined category mapping |
| Merchant lookup is a curated dictionary | New merchants need manual addition | Could be replaced with fuzzy matching + confidence |

## How to Improve

1. **Add more training data**: Edit `classifier/train.py` TRAINING_DATA with new labeled examples
2. **Add merchants**: Edit `classifier/merchant_lookup.py` MERCHANT_DB
3. **Retrain**: Click "Retrain Model" in the app sidebar, or delete models/*.joblib and restart
4. **Test on new bank**: Run `python scripts/evaluate.py your_new_statement.pdf` and review output
