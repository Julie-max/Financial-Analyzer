"""
User Feedback Store

Collects and persists user corrections for:
1. Category corrections — user fixes a misclassified transaction
2. New merchant mappings — user teaches the system a new merchant → category
3. CRF training examples — (future) user corrects payee extraction

Stored as JSON in data/feedback/:
- corrections.json  — category corrections (description → correct_category)
- merchants.json    — user-added merchant mappings (fragment → (name, category))

On retrain, these are merged with the base training data to improve the model.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FEEDBACK_DIR = Path(__file__).parent.parent / "data" / "feedback"
CORRECTIONS_FILE = FEEDBACK_DIR / "corrections.json"
MERCHANTS_FILE = FEEDBACK_DIR / "merchants.json"


def _ensure_dir():
    """Create feedback directory if it doesn't exist."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Category Corrections
# ---------------------------------------------------------------------------

def load_corrections() -> List[Dict]:
    """
    Load all category corrections.
    
    Returns list of dicts:
        {"description": str, "raw_description": str, "category": str,
         "original_category": str, "timestamp": str}
    """
    if not CORRECTIONS_FILE.exists():
        return []
    try:
        with open(CORRECTIONS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_correction(
    description: str,
    correct_category: str,
    original_category: str = "",
    raw_description: str = "",
) -> None:
    """
    Save a single category correction.
    If the same description already has a correction, update it.
    """
    _ensure_dir()
    corrections = load_corrections()
    
    # Update existing or append new
    found = False
    for c in corrections:
        if c["description"] == description:
            c["category"] = correct_category
            c["original_category"] = original_category
            c["raw_description"] = raw_description
            c["timestamp"] = datetime.now().isoformat()
            found = True
            break
    
    if not found:
        corrections.append({
            "description": description,
            "raw_description": raw_description,
            "category": correct_category,
            "original_category": original_category,
            "timestamp": datetime.now().isoformat(),
        })
    
    with open(CORRECTIONS_FILE, "w") as f:
        json.dump(corrections, f, indent=2)
    
    logger.info(f"Saved correction: '{description}' → {correct_category}")


def save_corrections_batch(corrections_list: List[Dict]) -> int:
    """
    Save multiple corrections at once.
    Each dict must have: description, category
    Optional: raw_description, original_category
    
    Returns number of corrections saved.
    """
    _ensure_dir()
    existing = load_corrections()
    existing_descs = {c["description"] for c in existing}
    
    count = 0
    for item in corrections_list:
        desc = item.get("description", "")
        cat = item.get("category", "")
        if not desc or not cat:
            continue
        
        if desc in existing_descs:
            # Update
            for c in existing:
                if c["description"] == desc:
                    c["category"] = cat
                    c["timestamp"] = datetime.now().isoformat()
                    break
        else:
            existing.append({
                "description": desc,
                "raw_description": item.get("raw_description", ""),
                "category": cat,
                "original_category": item.get("original_category", ""),
                "timestamp": datetime.now().isoformat(),
            })
        count += 1
    
    with open(CORRECTIONS_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    
    return count


def get_corrections_as_training_data() -> List[Tuple[str, str]]:
    """
    Convert corrections into training data format: [(description, category), ...]
    Includes VPA signal from raw_description if available.
    """
    from classifier.predict import extract_vpa_from_raw, classify_vpa
    
    corrections = load_corrections()
    training_pairs = []
    
    for c in corrections:
        desc = c["description"]
        raw = c.get("raw_description", "")
        category = c["category"]
        
        # Build feature text with VPA signal (same as predict.py)
        if raw:
            vpa = extract_vpa_from_raw(raw)
            vpa_type = classify_vpa(vpa)
            if vpa_type == "personal":
                desc = f"{desc} VPA_PERSONAL"
            elif vpa_type == "merchant":
                desc = f"{desc} VPA_MERCHANT"
            elif vpa_type == "qr":
                desc = f"{desc} VPA_QR"
        
        training_pairs.append((desc, category))
    
    return training_pairs


def clear_corrections() -> None:
    """Delete all corrections."""
    if CORRECTIONS_FILE.exists():
        CORRECTIONS_FILE.unlink()


# ---------------------------------------------------------------------------
# Merchant Mappings
# ---------------------------------------------------------------------------

def load_user_merchants() -> Dict[str, Tuple[str, str]]:
    """
    Load user-added merchant mappings.
    
    Returns dict: {"fragment": ("Clean Name", "Category")}
    """
    if not MERCHANTS_FILE.exists():
        return {}
    try:
        with open(MERCHANTS_FILE) as f:
            data = json.load(f)
        # Convert lists back to tuples
        return {k: tuple(v) for k, v in data.items()}
    except (json.JSONDecodeError, IOError):
        return {}


def save_merchant(
    fragment: str,
    clean_name: str,
    category: str,
) -> None:
    """
    Add or update a merchant mapping.
    
    Args:
        fragment: The text to match (substring, case-insensitive). e.g. "brindha"
        clean_name: Display name. e.g. "Brindha Cafe"
        category: Category. e.g. "Food"
    """
    _ensure_dir()
    merchants = load_user_merchants()
    merchants[fragment.lower().strip()] = (clean_name, category)
    
    # Convert tuples to lists for JSON
    with open(MERCHANTS_FILE, "w") as f:
        json.dump({k: list(v) for k, v in merchants.items()}, f, indent=2)
    
    logger.info(f"Saved merchant: '{fragment}' → ({clean_name}, {category})")


def delete_merchant(fragment: str) -> bool:
    """Remove a merchant mapping. Returns True if found and deleted."""
    merchants = load_user_merchants()
    key = fragment.lower().strip()
    if key in merchants:
        del merchants[key]
        _ensure_dir()
        with open(MERCHANTS_FILE, "w") as f:
            json.dump({k: list(v) for k, v in merchants.items()}, f, indent=2)
        return True
    return False


def clear_merchants() -> None:
    """Delete all user merchants."""
    if MERCHANTS_FILE.exists():
        MERCHANTS_FILE.unlink()


# ---------------------------------------------------------------------------
# Retrain with feedback
# ---------------------------------------------------------------------------

def retrain_with_feedback() -> Dict:
    """
    Retrain the classifier incorporating user feedback.
    
    Merges:
    1. Base TRAINING_DATA from train.py
    2. User corrections (converted to training pairs)
    3. User merchant additions are handled at lookup level (no retrain needed)
    
    Returns training metrics dict.
    """
    from classifier.train import TRAINING_DATA, train
    
    # Merge base + corrections
    user_corrections = get_corrections_as_training_data()
    merged_data = list(TRAINING_DATA) + user_corrections
    
    logger.info(
        f"Retraining: {len(TRAINING_DATA)} base + {len(user_corrections)} corrections "
        f"= {len(merged_data)} total"
    )
    
    # Retrain
    metrics = train(data=merged_data)
    metrics["user_corrections"] = len(user_corrections)
    return metrics


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_feedback_stats() -> Dict:
    """Get summary stats about stored feedback."""
    corrections = load_corrections()
    merchants = load_user_merchants()
    
    return {
        "total_corrections": len(corrections),
        "total_merchants": len(merchants),
        "corrections_by_category": _count_by_category(corrections),
        "merchants_by_category": _count_merchants_by_category(merchants),
    }


def _count_by_category(corrections: List[Dict]) -> Dict[str, int]:
    counts = {}
    for c in corrections:
        cat = c.get("category", "Unknown")
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def _count_merchants_by_category(merchants: Dict) -> Dict[str, int]:
    counts = {}
    for _, (_, cat) in merchants.items():
        counts[cat] = counts.get(cat, 0) + 1
    return counts
