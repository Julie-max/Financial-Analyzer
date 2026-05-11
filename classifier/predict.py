"""
Transaction Classifier — Two-Stage Prediction

Stage 1: Merchant lookup (known merchants → instant category)
Stage 2: Character n-gram TF-IDF + LinearSVC (everything else)

Stage 1 handles truncated UPI names like BOOKMYSH, MAKEMYTR.
Stage 2 handles everything else using character-level similarity.
"""

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd

from .train import preprocess_text, CATEGORIES, train
from .merchant_lookup import MerchantLookup

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "classifier_pipeline.joblib"


class TransactionClassifier:
    """
    Two-stage transaction classifier.

    Stage 1: MerchantLookup — instant category for known merchants.
    Stage 2: Character n-gram TF-IDF + LinearSVC — for unknown merchants.
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

    def predict(self, description: str) -> str:
        """Classify a single transaction description."""
        # Stage 1: merchant lookup
        result = self.merchant_lookup.lookup(description)
        if result is not None:
            _, category = result
            logger.debug(f"Lookup hit: '{description}' → {category}")
            return category

        # Stage 2: ML classifier
        processed = preprocess_text(description)
        label = self.pipeline.predict([processed])[0]
        return str(label)

    def predict_batch(self, descriptions: List[str]) -> List[str]:
        """Classify a list of transaction descriptions."""
        results = []
        ml_indices = []
        ml_descriptions = []

        # Stage 1: lookup pass
        for i, desc in enumerate(descriptions):
            result = self.merchant_lookup.lookup(desc)
            if result is not None:
                _, category = result
                results.append(category)
            else:
                results.append(None)  # placeholder
                ml_indices.append(i)
                ml_descriptions.append(preprocess_text(desc))

        # Stage 2: ML pass for unresolved
        if ml_descriptions:
            ml_labels = self.pipeline.predict(ml_descriptions)
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
        df["category"] = self.predict_batch(df["description"].tolist())
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
