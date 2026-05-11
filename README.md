# 💰 Financial Transaction Analyzer

A fully offline bank statement analysis system. Upload any bank statement PDF — the system extracts transactions, classifies spending, and visualizes your finances.

**No LLM. No neural network. No internet required.**

## Architecture

```
Bank Statement PDF
        │
        ▼
pdfplumber — spatial table extraction
        │  raw table rows
        ▼
CRF Parser — sequence labeling (token → PAYEE / REF_NUM / BANK / VPA / ...)
        │  clean merchant names
        ▼
Post-Processor — date normalization, deduplication, validation
        │  clean DataFrame
        ▼
Two-Stage Classifier:
  Stage 1: Merchant lookup (known merchants → instant category)
  Stage 2: Char n-gram TF-IDF + LinearSVC (everything else)
        │  categorized DataFrame
        ▼
Streamlit UI — charts, insights, CSV export
```

## ML Algorithms Used

### Parsing: CRF (Conditional Random Fields)
The classic pre-deep-learning NLP algorithm for sequence labeling. Given a tokenized transaction description, the CRF labels each token as `PAYEE`, `REF_NUM`, `BANK`, `VPA`, `PREFIX`, `SUFFIX`, etc. — then extracts the `PAYEE` tokens as the merchant name.

This generalizes across all bank formats (UPI, NEFT, IMPS, BIL, ATM, POS) because it learns structural patterns from labeled examples, not bank-specific rules.

### Classification: Character n-gram TF-IDF + LinearSVC
Uses character-level n-grams (2–5 chars) instead of word n-grams. This handles truncated UPI merchant names like `BOOKMYSH` (BookMyShow), `MAKEMYTR` (MakeMyTrip), `SWIGGYSTOR` (Swiggy) — because character sequences overlap with the full names in training data.

A two-stage approach:
1. **Merchant lookup** — instant category for ~120 known merchants
2. **LinearSVC** — handles everything else using character similarity

## Project Structure

```
financial-analyzer/
├── app/
│   └── streamlit_app.py          # Streamlit UI
├── classifier/
│   ├── merchant_lookup.py        # Stage 1: known merchant database
│   ├── predict.py                # Two-stage classifier
│   └── train.py                  # Char TF-IDF + LinearSVC training
├── data/
│   ├── generate_sample_pdf.py    # Generate test PDFs (HDFC + SBI)
│   ├── hdfc_sample_statement.pdf
│   └── sbi_sample_statement.pdf
├── models/
│   ├── classifier_pipeline.joblib  # Trained TF-IDF + LinearSVC
│   ├── crf_parser.joblib           # Trained CRF model
│   └── categories.json
├── parser/
│   ├── crf_parser.py             # CRF sequence labeler for descriptions
│   ├── pdf_extractor.py          # pdfplumber + PyMuPDF fallback
│   ├── post_processor.py         # Date normalization, dedup, validation
│   └── table_parser.py           # Column detection, row parsing, schema detection
├── utils/
│   ├── insights.py               # Analytics and natural language insights
│   └── pipeline.py               # End-to-end orchestration
├── requirements.txt
├── setup.py
└── README.md
```

## Setup

### 1. Python 3.10+

```bash
python --version
```

### 2. Create and activate a virtual environment

```bash
cd financial-analyzer
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Run setup (installs dependencies + trains models)

```bash
python setup.py
```

This installs all dependencies, trains the spend classifier, and trains the CRF parser.

### 4. Start the app

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501

## Supported Bank Formats

The system handles any digital (non-scanned) bank statement PDF. Tested on:

| Bank | Format | Notes |
|------|--------|-------|
| ICICI | UPI/DR/REF/PAYEE/BANK/VPA | Standard ICICI detailed statement |
| SBI | WDL TFR UPI/DR/... AT BRANCH | SBI passbook format |
| City Union Bank | TO ONL UPI/DR/... U::00116 | CUB mPassbook format |
| HDFC | Debit/Credit columns | Standard HDFC statement |
| Any other | CRF generalizes | No bank-specific code needed |

## Spending Categories

| Category | Examples |
|----------|---------|
| Food | Swiggy, Zomato, restaurants, groceries, cafes |
| Travel | Uber, IRCTC, flights, fuel, hotels, cabs |
| Shopping | Amazon, Flipkart, clothing, electronics |
| Bills | Electricity, mobile, subscriptions, EMIs, insurance |
| Entertainment | Movies, gaming, concerts, streaming |
| Others | Salary, transfers, investments, ATM |

## Why These Algorithms

### Why CRF for parsing?
Bank transaction descriptions are structured sequences — `UPI/DR/REFNO/PAYEE/BANK/VPA`. CRF is the right tool for sequence labeling: it considers the context of neighboring tokens, not just individual tokens in isolation. A new bank format with the same structural logic is handled correctly without any code changes.

### Why character n-gram TF-IDF + LinearSVC for classification?
UPI truncates merchant names to 8 characters: `BOOKMYSH`, `MAKEMYTR`, `SWIGGYSTOR`. Word-level TF-IDF fails on these because there's zero word overlap with the training data. Character n-grams (2–5 chars) capture partial matches: `BOOKMYSH` shares `BOOK`, `OOKM`, `OKMY`, `KMYS`, `MYSH` with `BOOKMYSHOW`. LinearSVC is fast, lightweight, and works well with high-dimensional sparse character features.

### Why not LLM?
- Financial data must stay on-device — no API calls
- LLMs are slow (20–60s per PDF) and non-deterministic
- The parsing problem is structured, not semantic — CRF is the right tool
- The classification problem is keyword-driven — character TF-IDF is sufficient

## Known Limitations

| Limitation | Workaround |
|-----------|------------|
| Scanned/image PDFs | Convert with OCR (Tesseract) first |
| Password-protected PDFs | Unlock PDF before uploading |
| Multi-currency statements | Amounts treated as single currency |

## Troubleshooting

**No transactions extracted:**
- Ensure the PDF is a digital (text-based) bank statement, not a scan
- Try a different PDF viewer to confirm text is selectable

**Wrong categories:**
- Click "Retrain Model" in the sidebar after adding training examples to `classifier/train.py`
- Add the merchant to `classifier/merchant_lookup.py` for instant lookup
