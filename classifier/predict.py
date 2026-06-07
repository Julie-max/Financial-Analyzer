"""
Transaction Classifier — Unified Single-Stage ML

Architecture:
  1. Feature extraction from raw description (preserves VPA + structural signals)
  2. Character n-gram TF-IDF + LinearSVC classification

Key insight: The raw bank description contains VPA info that strongly signals
whether a payment is personal (P2P) vs merchant. By preserving this in the
feature text, the classifier learns these patterns directly — no separate
VPA classifier needed.

Personal payment signals (→ Others):
  - VPA is a phone number: 9345785876, 8142996999
  - VPA starts with 'q' + digits: q721834704, q854647665
  - VPA is a name: julieschan, mchandrasekara, parameswariraj
  - Payee is a short name: MR KALID, SANTHI S, C BRIANT

Merchant payment signals (→ specific category):
  - VPA contains brand: swiggystores, zomato, makemytrip, indianrail
  - VPA has merchant patterns: paytmqr*, vyapar.*, gpay-*, bharatpe*
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd

from .train import preprocess_text, CATEGORIES, train
from .merchant_lookup import MerchantLookup

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "classifier_pipeline.joblib"


# ---------------------------------------------------------------------------
# VPA pattern detection (replaces the separate VPA classifier)
# ---------------------------------------------------------------------------

# Phone number pattern (Indian 10-digit)
_PHONE_RE = re.compile(r'^\d{10}$')
# q-prefix personal UPI (PhonePe/GPay personal)
_Q_PERSONAL_RE = re.compile(r'^q\d{6,}$', re.IGNORECASE)
# Paytm QR codes (small vendors — treat as merchant/qr)
_PAYTM_QR_RE = re.compile(r'^paytmqr|^paytm[\.\-]?[sd]\d|^ptm[a-f0-9]{10,}', re.IGNORECASE)
# Known merchant VPA patterns
_MERCHANT_VPA_RE = re.compile(
    r'(swiggystores|payzomato|zomato|makemytrip|indianrail|redbus|'
    r'jiofiberpre|airtel|spotify|linkedin|zerodha|amazon|flipkart|'
    r'bookmyshow|bigtree|amazonaws|billdeskpg|sbicardsand|'
    r'eurekaforbes|bhimcashback|playstore|gpayutili|gpayrechar|'
    r'zeptomarke|brindhacaf|fellowkann)',
    re.IGNORECASE
)
# GPay merchant codes
_GPAY_MERCHANT_RE = re.compile(r'^gpay[\-]?\d{5,}', re.IGNORECASE)
# Vyapar (merchant POS)
_VYAPAR_RE = re.compile(r'^vyapar\.', re.IGNORECASE)
# BharatPe / POS
_BHARATPE_RE = re.compile(r'^bharatpe|^pos\.\d|^mab\.\d', re.IGNORECASE)
# Personal name VPAs (lowercase alpha, possibly with trailing digits)
_NAME_VPA_RE = re.compile(r'^[a-z]{4,}[a-z0-9]*$', re.IGNORECASE)


def classify_vpa(vpa: str) -> str:
    """
    Classify a VPA string as 'personal', 'merchant', 'qr', or 'unknown'.
    Uses pattern matching — deterministic and interpretable.
    """
    if not vpa or not vpa.strip():
        return "unknown"
    
    vpa = vpa.strip().lower()
    
    # Strip @domain if present
    if '@' in vpa:
        vpa = vpa.split('@')[0]
    
    # Phone numbers → personal
    if _PHONE_RE.match(vpa):
        return "personal"
    
    # q-prefix + digits → personal (PhonePe/GPay P2P)
    if _Q_PERSONAL_RE.match(vpa):
        return "personal"
    
    # Known merchant VPAs
    if _MERCHANT_VPA_RE.search(vpa):
        return "merchant"
    
    # GPay merchant codes (gpay-11265, gpay112517363)
    if _GPAY_MERCHANT_RE.match(vpa):
        return "merchant"
    
    # Vyapar POS terminals
    if _VYAPAR_RE.match(vpa):
        return "merchant"
    
    # BharatPe / POS / MAB
    if _BHARATPE_RE.match(vpa):
        return "qr"
    
    # Paytm QR codes
    if _PAYTM_QR_RE.match(vpa):
        return "qr"
    
    # Paytm with short alphanumeric suffix (paytm.s1zo, paytm-8927)
    if re.match(r'^paytm', vpa, re.IGNORECASE):
        return "qr"
    
    # Name-like VPAs (mostly alpha, 4+ chars) → personal
    if _NAME_VPA_RE.match(vpa) and not any(
        brand in vpa for brand in [
            'swiggy', 'zomato', 'amazon', 'flipkart', 'uber', 'ola',
            'airtel', 'jio', 'google', 'apple', 'netflix', 'spotify',
            'zerodha', 'groww', 'irctc', 'redbus', 'makemytrip',
            'bookmyshow', 'phonepe', 'paytm', 'blinkit', 'zepto',
        ]
    ):
        return "personal"
    
    return "unknown"


def extract_vpa_from_raw(raw_description: str) -> str:
    """
    Extract VPA from a raw transaction description.
    Works across SBI, ICICI, CUB, Axis formats.
    
    Format patterns:
    - SBI: WDL TFR UPI/DR/REF/PAYEE/BANK/VPA/trailing...
    - CUB: UPI/DR/REF/PAYEE/BANK/VPA/trailing...
    - ICICI: UPI/PAYEE/VPA/REMARK/BANK/REF/trailing...
    - NEFT: NEFT-REF-PAYEE-REMARK-... (no VPA)
    """
    if not raw_description:
        return ""
    
    # Normalize whitespace/newlines
    raw = " ".join(raw_description.split())
    
    # NEFT/BIL/MMT/INF formats don't have VPA
    if any(raw.upper().startswith(p) for p in ['NEFT-', 'BIL/', 'MMT/', 'INF/', 'IPS/', 'VPS/']):
        return ""
    
    parts = raw.split('/')
    if len(parts) < 4:
        return ""
    
    # Known bank codes (used to identify bank position)
    bank_codes = {
        'YESB', 'HDFC', 'SBIN', 'UTIB', 'IOBA', 'UBIN', 'KKBK',
        'CNRB', 'BARB', 'FDRL', 'IDIB', 'MAHB', 'COSB', 'BKID',
        'SRCB', 'UNBA', 'ICIC', 'AXIS', 'TMBL', 'CIUB', 'IDFB',
        'AIRP', 'INDB', 'PUNB', 'CBIN', 'IBKL', 'NSPB',
    }
    
    # Detect format by looking for bank code position
    # SBI/CUB format: ...PAYEE/BANK(4-char)/VPA/...
    # The VPA is right after the bank code
    for i, part in enumerate(parts):
        part_clean = part.strip().upper().replace(' ', '')
        # Check if this part starts with a bank code (handles "YESB", "Y ESB" etc)
        # Also handle "YES BANK L", "ICICI Bank", etc (ICICI format)
        if part_clean in bank_codes or (len(part_clean) == 4 and part_clean in bank_codes):
            # VPA is the next part
            if i + 1 < len(parts):
                vpa_candidate = parts[i + 1].strip()
                # Clean up: remove trailing "UPI ...", "Paym ...", "Mand ..." etc
                vpa_candidate = re.split(r'\s+(UPI|Paym|Mand|Firs)\b', vpa_candidate)[0].strip()
                if vpa_candidate and len(vpa_candidate) >= 4 and not vpa_candidate.isdigit():
                    return vpa_candidate
    
    # ICICI format: UPI/PAYEE/VPA/REMARK/BANK NAME/REF/...
    # VPA is in position 2 (0-indexed) — after "UPI" prefix and payee
    if parts[0].strip().upper() == 'UPI' and len(parts) >= 5:
        # Position 2 should be VPA in ICICI format
        vpa_candidate = parts[2].strip()
        # Clean spaces that come from PDF extraction
        vpa_candidate = vpa_candidate.replace(' ', '')
        # Validate it looks like a VPA (contains @, or is alphanumeric with dots/dashes)
        if '@' in vpa_candidate:
            return vpa_candidate.split('@')[0]
        if re.match(r'^[a-z0-9][\w.\-]{3,}$', vpa_candidate, re.IGNORECASE):
            digit_ratio = sum(c.isdigit() for c in vpa_candidate) / len(vpa_candidate)
            # Skip if it's mostly digits (it's a ref number, not VPA)
            if digit_ratio < 0.7:
                return vpa_candidate
    
    return ""


class TransactionClassifier:
    """
    Unified transaction classifier.
    
    Uses the raw description to extract VPA signals, then combines
    payee name + VPA type as features for the ML classifier.
    
    Merchant lookup handles known merchants for instant classification.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        self.model_path = Path(model_path)
        self.pipeline = None
        self.merchant_lookup = MerchantLookup()
        self._load_or_train()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, description: str, raw_description: str = "") -> str:
        """Classify a single transaction description."""
        # Stage 1: merchant lookup on both clean and raw description
        result = self.merchant_lookup.lookup(description)
        if result is not None:
            _, category = result
            return category
        
        if raw_description:
            result = self.merchant_lookup.lookup(raw_description)
            if result is not None:
                _, category = result
                return category

        # Stage 2: ML classifier with VPA signal
        feature_text = self._build_feature_text(description, raw_description)
        processed = preprocess_text(feature_text)
        label = self.pipeline.predict([processed])[0]
        return str(label)

    def predict_batch(self, descriptions: List[str], raw_descriptions: List[str] = None) -> List[str]:
        """Classify a list of transaction descriptions."""
        if raw_descriptions is None:
            raw_descriptions = [""] * len(descriptions)
        
        results = []
        ml_indices = []
        ml_features = []

        # Stage 1: lookup pass
        for i, (desc, raw) in enumerate(zip(descriptions, raw_descriptions)):
            result = self.merchant_lookup.lookup(desc)
            if result is None and raw:
                result = self.merchant_lookup.lookup(raw)
            
            if result is not None:
                _, category = result
                results.append(category)
            else:
                results.append(None)  # placeholder
                ml_indices.append(i)
                feature_text = self._build_feature_text(desc, raw)
                ml_features.append(preprocess_text(feature_text))

        # Stage 2: ML pass for unresolved
        if ml_features:
            ml_labels = self.pipeline.predict(ml_features)
            for idx, label in zip(ml_indices, ml_labels):
                results[idx] = str(label)

        lookup_count = len(descriptions) - len(ml_indices)
        ml_count = len(ml_indices)
        logger.info(
            f"Classification: {lookup_count} via lookup, {ml_count} via ML"
        )

        return results

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a 'category' column to a transactions DataFrame."""
        if df.empty:
            df["category"] = pd.Series(dtype=str)
            return df
        df = df.copy()
        
        descriptions = df["description"].tolist()
        raw_descriptions = df["raw_description"].tolist() if "raw_description" in df.columns else [""] * len(descriptions)
        
        df["category"] = self.predict_batch(descriptions, raw_descriptions)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_feature_text(self, description: str, raw_description: str = "") -> str:
        """
        Build the feature string for the classifier.
        Combines description + VPA-derived signal.
        """
        # Extract VPA from raw description
        vpa = extract_vpa_from_raw(raw_description) if raw_description else ""
        vpa_type = classify_vpa(vpa)
        
        # Append VPA signal to description for classifier
        if vpa_type == "personal":
            return f"{description} VPA_PERSONAL"
        elif vpa_type == "merchant":
            return f"{description} VPA_MERCHANT"
        elif vpa_type == "qr":
            return f"{description} VPA_QR"
        
        return description

    def _load_or_train(self):
        """Load model from disk, or train a fresh one if not found."""
        if self.model_path.exists():
            logger.info(f"Loading classifier from {self.model_path}")
            self.pipeline = joblib.load(self.model_path)
            logger.info("Classifier loaded successfully.")
        else:
            logger.warning(
                f"Model not found at {self.model_path}. Training a new model..."
            )
            train(model_dir=self.model_path.parent)
            self.pipeline = joblib.load(self.model_path)
            logger.info("New model trained and loaded.")


def classify_transactions(df: pd.DataFrame, model_path: str = None) -> pd.DataFrame:
    """Convenience function."""
    classifier = TransactionClassifier(model_path=model_path)
    return classifier.classify_dataframe(df)
