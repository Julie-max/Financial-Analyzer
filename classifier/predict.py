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

        # Stage 2: ML classifier with VPA signal
        feature_text = _build_feature_text(description)
        processed = preprocess_text(feature_text)
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
                ml_descriptions.append(preprocess_text(_build_feature_text(desc)))

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


def _build_feature_text(description: str) -> str:
    """
    Build the feature string for the classifier by combining the description
    with a VPA-derived signal from the ML VPA classifier.
    """
    try:
        from parser.crf_parser import get_parser
        _, vpa = get_parser().extract_payee_and_vpa(description)
        vpa_signal = get_vpa_classifier().predict(vpa)
        if vpa_signal and vpa_signal != "unknown":
            return f"{description} VPA_{vpa_signal.upper()}"
    except Exception:
        pass
    return description


# ---------------------------------------------------------------------------
# VPA Classifier — ML-based, no keyword lists
# ---------------------------------------------------------------------------

_vpa_classifier_instance = None

def get_vpa_classifier():
    """Get or create the singleton VPA classifier."""
    global _vpa_classifier_instance
    if _vpa_classifier_instance is None:
        _vpa_classifier_instance = VPAClassifier()
    return _vpa_classifier_instance


class VPAClassifier:
    """
    ML classifier for UPI VPA strings.
    Predicts: 'personal' | 'merchant' | 'qr' | 'unknown'

    Uses character n-gram TF-IDF + LinearSVC.
    Trained on real VPA examples from Indian bank statements.

    Why ML instead of rules:
    - VPA formats vary across banks and payment apps
    - Character n-grams capture structural patterns:
      mariabritto281 → mostly alpha + trailing digits → personal
      swiggystores@icici → brand@domain → merchant
      paytmqr6x5 → qr prefix + alphanumeric → qr
    - Generalizes to unseen VPAs through character similarity
    """

    MODEL_PATH = Path(__file__).parent.parent / "models" / "vpa_classifier.joblib"

    # Training data: (vpa_string, label)
    # personal = P2P payment to an individual
    # merchant = payment to a business/brand
    # qr       = QR code payment (small vendor, ambiguous)
    # unknown  = no VPA available
    TRAINING_DATA = [
        # ── personal ──────────────────────────────────────────────────────
        ("mariabritto281", "personal"),
        ("julieschandra", "personal"),
        ("julieschan", "personal"),
        ("jeyanthim2", "personal"),
        ("ciceeliachandr", "personal"),
        ("sanjai1710", "personal"),
        ("sanjayprak", "personal"),
        ("suryaprasa", "personal"),
        ("annrufina6", "personal"),
        ("ksmahaling", "personal"),
        ("jeyanthim2", "personal"),
        ("briantjulian", "personal"),
        ("aliappas41", "personal"),
        ("ambavaramr", "personal"),
        ("paulprem19", "personal"),
        ("mwaran0623", "personal"),
        ("xalxorosma", "personal"),
        ("moorthyixm", "personal"),
        ("sivas123", "personal"),
        ("9791383910", "personal"),
        ("8142996999", "personal"),
        ("9344942103", "personal"),
        ("9939431812", "personal"),
        ("9842174135", "personal"),
        ("7218009639", "personal"),
        ("8904399165", "personal"),
        ("9345785876", "personal"),
        ("6383456381", "personal"),
        ("7507675269", "personal"),
        ("q854647665", "personal"),
        ("q319459213", "personal"),
        ("q469385175", "personal"),
        ("q673607959", "personal"),
        ("q682399814", "personal"),
        ("q721834704", "personal"),
        ("q757054153", "personal"),
        ("q987029173", "personal"),
        ("q188105277", "personal"),
        ("q553307949", "personal"),
        ("q619306841", "personal"),
        ("q928397193", "personal"),
        ("q965550772", "personal"),
        ("q580878242", "personal"),
        ("q415786617", "personal"),
        ("q845856751", "personal"),
        ("q442640661", "personal"),
        ("q699995027", "personal"),
        ("q764569048", "personal"),
        ("q948252874", "personal"),
        ("q042727764", "personal"),
        ("q325468723", "personal"),
        ("q827685647", "personal"),
        ("q689897013", "personal"),
        ("q806521597", "personal"),
        ("q704219312", "personal"),
        ("q080693189", "personal"),
        ("q220475914", "personal"),
        ("q421572316", "personal"),
        ("q450139409", "personal"),
        ("q089516741", "personal"),
        ("q154483475", "personal"),
        ("q211469701", "personal"),
        ("q053976079", "personal"),
        ("q913260809", "personal"),
        ("q775953742", "personal"),
        ("q594434143", "personal"),
        ("q015282577", "personal"),
        ("q558456400", "personal"),
        ("q667408334", "personal"),
        ("q083062573", "personal"),
        ("q950375434", "personal"),
        ("q721834704", "personal"),
        ("q708912994", "personal"),
        ("q129119317", "personal"),
        ("q653888387", "personal"),
        ("q703455802", "personal"),
        ("q929119317", "personal"),
        ("q668026133", "personal"),
        ("q553307949", "personal"),
        ("q188105277", "personal"),
        ("q619306841", "personal"),
        ("q987029173", "personal"),
        ("q721834704", "personal"),
        ("b95144323", "personal"),
        ("9503604016", "personal"),
        ("7350835305", "personal"),
        ("8767040864", "personal"),
        ("9860542837", "personal"),
        ("9322818098", "personal"),
        ("8888125253", "personal"),
        ("9548753590", "personal"),
        ("9042970059", "personal"),
        ("9360279266", "personal"),
        ("9360943929", "personal"),
        ("9884380041", "personal"),
        ("9443619185", "personal"),
        ("9443765569", "personal"),
        ("9600428400", "personal"),
        ("9940770900", "personal"),
        ("8682866654", "personal"),
        ("8310297412", "personal"),
        ("9101747007", "personal"),
        ("7411902872", "personal"),
        ("8015912245", "personal"),
        ("8525888517", "personal"),
        ("9345224453", "personal"),
        ("8940458895", "personal"),
        ("8122853440", "personal"),
        ("8951912199", "personal"),
        ("8610008380", "personal"),
        ("9585667175", "personal"),
        ("9600428400", "personal"),
        ("8754890671", "personal"),
        ("9881994726", "personal"),
        ("9503604016", "personal"),
        ("9042970059", "personal"),
        ("fellowkann", "personal"),
        ("annrufina6", "personal"),
        ("bala979165", "personal"),
        ("seeni4414", "personal"),
        ("santhoraj8", "personal"),
        ("drsulopaul", "personal"),
        ("paulprem19", "personal"),
        ("mvasanthip", "personal"),
        ("suryaprasa", "personal"),
        ("firozsheik", "personal"),
        ("rameshwarg", "personal"),
        ("srisridara", "personal"),
        ("yadavshivm", "personal"),
        ("manojjadha", "personal"),
        ("vishalpaik", "personal"),
        ("akashkale3", "personal"),
        ("prbhakaram", "personal"),
        ("imranshaik", "personal"),
        ("revatibork", "personal"),
        ("dhannanak", "personal"),
        ("gurdeet", "personal"),
        ("esakkimuth", "personal"),
        ("mahadulera", "personal"),
        ("bajajpay6", "personal"),
        ("santosh", "personal"),
        ("chouthma", "personal"),
        ("mahesha", "personal"),
        ("fathimab", "personal"),
        ("periyasa", "personal"),
        ("meena", "personal"),
        ("mariya", "personal"),
        ("satheesh", "personal"),
        ("mohanraj", "personal"),
        ("dinesh", "personal"),
        ("vinod", "personal"),
        ("rupali", "personal"),
        ("sujit", "personal"),
        ("vishal", "personal"),
        ("ashish", "personal"),
        ("pavan", "personal"),
        ("rajaram", "personal"),
        ("balaji", "personal"),
        ("navanath", "personal"),
        ("kalyana", "personal"),
        ("vetri", "personal"),
        ("rdevi", "personal"),
        ("yuvaraja", "personal"),
        ("susheela", "personal"),
        ("mamatha", "personal"),
        ("yoganand", "personal"),
        ("s kumare", "personal"),
        ("rohan ga", "personal"),
        ("kannayir", "personal"),
        ("s appas", "personal"),
        ("mr moham", "personal"),
        ("atria co", "personal"),
        ("mr subra", "personal"),
        ("sree var", "personal"),
        ("m gunase", "personal"),
        ("chai fi", "personal"),
        ("mr balam", "personal"),
        ("mrs indr", "personal"),
        ("pavithra", "personal"),
        ("anantha", "personal"),
        ("diwa ent", "personal"),
        ("nagaraja", "personal"),
        ("suryapra", "personal"),
        ("chandru", "personal"),
        ("muneeswa", "personal"),
        ("gurusamy", "personal"),
        ("alagarpr", "personal"),
        ("sulochan", "personal"),
        ("vasanthi", "personal"),
        ("dhaksina", "personal"),
        ("mohanbab", "personal"),
        ("pandimun", "personal"),
        ("annapura", "personal"),
        ("rosma x", "personal"),
        ("gideon s", "personal"),
        ("krishnar", "personal"),
        ("premkuma", "personal"),
        ("shafeeh", "personal"),
        ("mr mahal", "personal"),
        ("v narend", "personal"),
        ("suresh t", "personal"),
        ("juliesch", "personal"),
        ("meena lo", "personal"),
        ("ekart", "personal"),
        ("shri man", "personal"),
        ("chouthma", "personal"),
        ("mahesha", "personal"),
        ("fathimab", "personal"),
        ("mr s ka", "personal"),
        ("ms annru", "personal"),
        ("autope p", "personal"),
        ("irctcautop", "personal"),

        # ── merchant ──────────────────────────────────────────────────────
        ("swiggystores@i", "merchant"),
        ("swiggystores@icici", "merchant"),
        ("payzomato@", "merchant"),
        ("zomatoorder@p", "merchant"),
        ("makemytrip", "merchant"),
        ("makemytrip@hdfc", "merchant"),
        ("redbus1onl", "merchant"),
        ("redbus32 r", "merchant"),
        ("redbus1 db", "merchant"),
        ("irctcautop", "merchant"),
        ("indianrail", "merchant"),
        ("jiofiberprepai", "merchant"),
        ("airtel-pre", "merchant"),
        ("spotifyindiall", "merchant"),
        ("linkedin.bdsi@", "merchant"),
        ("zerodhamf@hdfc", "merchant"),
        ("zerodhafundhou", "merchant"),
        ("amazonaws@rapl", "merchant"),
        ("amazon.refunds", "merchant"),
        ("gpayutility@o", "merchant"),
        ("gpayrechar", "merchant"),
        ("gpay-utili", "merchant"),
        ("playstore1", "merchant"),
        ("playstore@", "merchant"),
        ("bigtreeent", "merchant"),
        ("bookmyshow", "merchant"),
        ("paytm-8726", "merchant"),
        ("paytm-651536@p", "merchant"),
        ("paytm-53817591", "merchant"),
        ("travel1paytm@h", "merchant"),
        ("ekart@ybl", "merchant"),
        ("cbdt.payu@hdfc", "merchant"),
        ("sbiepay.ectnp1", "merchant"),
        ("tangedco@india", "merchant"),
        ("lifeincorpofin", "merchant"),
        ("appleservi", "merchant"),
        ("appleservices", "merchant"),
        ("amazon.refunds", "merchant"),
        ("bigtreeent", "merchant"),
        ("oyorooms585056", "merchant"),
        ("thirdwavecoffe", "merchant"),
        ("eurekaforbeslt", "merchant"),
        ("bhimcashback@h", "merchant"),
        ("goog-payme", "merchant"),
        ("googlepay", "merchant"),
        ("phonepe", "merchant"),
        ("upiswiggy@icic", "merchant"),
        ("sbicardsandpay", "merchant"),
        ("billdeskpg.sbi", "merchant"),
        ("chaifi25@f", "merchant"),
        ("lemon.tree.hot", "merchant"),
        ("vyapar.175", "merchant"),
        ("vyapar.170", "merchant"),
        ("vyapar.172", "merchant"),
        ("vyapar.173", "merchant"),
        ("actcorp1 p", "merchant"),
        ("annrufina6", "merchant"),
        ("fellowkann", "merchant"),
        ("jeyanthim2", "merchant"),
        ("gpay-11265", "merchant"),
        ("gpay-12190", "merchant"),
        ("gpay-11264", "merchant"),
        ("gpay-11256", "merchant"),
        ("gpay-11239", "merchant"),
        ("gpay-11263", "merchant"),
        ("gpay112517363", "merchant"),
        ("gpay112443093", "merchant"),
        ("gpay112528973", "merchant"),
        ("gpay121901993", "merchant"),
        ("gpay121964207", "merchant"),
        ("gpay112607908", "merchant"),

        # ── qr ────────────────────────────────────────────────────────────
        ("paytmqr6x5", "qr"),
        ("paytmqr6tt", "qr"),
        ("paytmqr6uj", "qr"),
        ("paytmqr6wx", "qr"),
        ("paytmqr6nd", "qr"),
        ("paytmqr6et", "qr"),
        ("paytmqr6lf", "qr"),
        ("paytmqr6fr", "qr"),
        ("paytmqr6p8", "qr"),
        ("paytmqr6q3", "qr"),
        ("paytmqr6yj", "qr"),
        ("paytmqr6ol", "qr"),
        ("paytmqr5ek", "qr"),
        ("paytmqr5d6", "qr"),
        ("paytmqr5ze", "qr"),
        ("paytmqr5dg", "qr"),
        ("paytmqr5vj", "qr"),
        ("paytmqr10p", "qr"),
        ("paytmqr1h9", "qr"),
        ("paytmqr1qullne", "qr"),
        ("paytmqr2810050", "qr"),
        ("paytmqr67s", "qr"),
        ("paytmqr674", "qr"),
        ("paytmqr6x5", "qr"),
        ("paytm s1du", "qr"),
        ("paytm s1mq", "qr"),
        ("paytm s1we", "qr"),
        ("paytm s1to", "qr"),
        ("paytm s1wh", "qr"),
        ("paytm s1rf", "qr"),
        ("paytm s1do", "qr"),
        ("paytm s1vj", "qr"),
        ("paytm s1tu", "qr"),
        ("paytm s1ts", "qr"),
        ("paytm s1zo", "qr"),
        ("paytm s1iv", "qr"),
        ("paytm s1h7", "qr"),
        ("paytm s1cx", "qr"),
        ("paytm s1sz", "qr"),
        ("paytm s1wx", "qr"),
        ("paytm s208", "qr"),
        ("paytm s207", "qr"),
        ("paytm s21t", "qr"),
        ("paytm s23b", "qr"),
        ("paytm s186", "qr"),
        ("paytm s233", "qr"),
        ("paytm d915", "qr"),
        ("paytm d192", "qr"),
        ("paytm d150", "qr"),
        ("paytm d153", "qr"),
        ("paytm d183", "qr"),
        ("paytm.s1zo", "qr"),
        ("paytm.s1iv", "qr"),
        ("paytm.s1w4", "qr"),
        ("paytm.s1dg", "qr"),
        ("paytm.s1mq", "qr"),
        ("paytm.s1du", "qr"),
        ("paytm.s1we", "qr"),
        ("paytm.s1wh", "qr"),
        ("paytm.s1rf", "qr"),
        ("paytm.s1do", "qr"),
        ("paytm.s1vj", "qr"),
        ("paytm.s1tu", "qr"),
        ("paytm.s1ts", "qr"),
        ("paytm.s1h7", "qr"),
        ("paytm.s1cx", "qr"),
        ("paytm.s1sz", "qr"),
        ("paytm.s1wx", "qr"),
        ("paytm.s1320g5", "qr"),
        ("paytm.s1jb7og", "qr"),
        ("paytm.s1k20rr", "qr"),
        ("paytm.s1lwt0j", "qr"),
        ("paytm.s1avb67", "qr"),
        ("paytm.s1b7og", "qr"),
        ("paytm.d0108406", "qr"),
        ("paytm.d7924730", "qr"),
        ("paytm.d9136422", "qr"),
        ("paytm8736701", "qr"),
        ("paytm8740474", "qr"),
        ("paytm83567411", "qr"),
        ("paytm-9645", "qr"),
        ("paytm.s2208", "qr"),
        ("paytmqr6x5", "qr"),
        ("mab.037347", "qr"),
        ("mab.037326", "qr"),
        ("mab.037348", "qr"),
        ("mab.0373450009", "qr"),
        ("mab.0373470122", "qr"),
        ("pos.113320", "qr"),
        ("bharatpe 9", "qr"),
        ("bharatpe 8", "qr"),
        ("stk-826289", "qr"),
        ("cf sribala", "qr"),
        ("j a w 7522", "qr"),
        ("6207261000", "qr"),
        ("630336370", "qr"),
        ("9500941833", "qr"),
        ("9791383910", "qr"),
        ("8142996999", "qr"),
        ("ptmb7ce7a510fad", "qr"),
        ("ptmd59a0f4f7d7e", "qr"),
        ("ptmc5258a4565ba", "qr"),
        ("ptmb3521d51ac05", "qr"),
        ("ptm36cd5cb89db4", "qr"),
        ("ptm56a2b24d84d", "qr"),
        ("ptm9876eed19c9", "qr"),
        ("ptm8af2cabfa5b", "qr"),
        ("ptm5d833a14763", "qr"),
        ("ptm6cab861873b", "qr"),
        ("ptm67c13b5f9f6", "qr"),
        ("ptm38972aee75d", "qr"),
        ("ptm405f67b2a32", "qr"),
        ("ptmedf6c8aee49", "qr"),
        ("ptmeeb8e106cc4", "qr"),
        ("ptm31eb4f9f4aa", "qr"),
        ("ptm6744a7049bf", "qr"),
        ("ptme02ac50a74a", "qr"),
        ("ptm96ae6f9cb84", "qr"),
        ("ptm71766f6440e", "qr"),
        ("ptmba421703932", "qr"),
        ("ptmf6e824ccdfcb", "qr"),
        ("ptm3d9d209153fc", "qr"),
        ("ptm262a3651d6db", "qr"),
        ("ptmb67e7e63476", "qr"),
        ("ptmc5258a4565ba", "qr"),
        ("pytm5090880405", "qr"),
        ("pytm5111780245", "qr"),
        ("pytm5112680532100", "qr"),
        ("pytm5121080625", "qr"),
    ]

    def __init__(self):
        self.pipeline = None
        self._load_or_train()

    def predict(self, vpa: str) -> str:
        """Predict VPA type: 'personal', 'merchant', 'qr', or 'unknown'."""
        if not vpa or not vpa.strip():
            return "unknown"
        return self.pipeline.predict([vpa.lower().strip()])[0]

    def _load_or_train(self):
        if self.MODEL_PATH.exists():
            logger.info(f"Loading VPA classifier from {self.MODEL_PATH}")
            self.pipeline = joblib.load(self.MODEL_PATH)
        else:
            logger.info("Training VPA classifier...")
            self._train()

    def _train(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.pipeline import Pipeline as SKPipeline

        X = [vpa.lower().strip() for vpa, _ in self.TRAINING_DATA]
        y = [label for _, label in self.TRAINING_DATA]

        self.pipeline = SKPipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                min_df=1,
                sublinear_tf=True,
            )),
            ("clf", LinearSVC(
                C=1.0,
                dual="auto",
                max_iter=2000,
                class_weight="balanced",
            )),
        ])
        self.pipeline.fit(X, y)

        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.MODEL_PATH)

        preds = self.pipeline.predict(X)
        correct = sum(p == t for p, t in zip(preds, y))
        logger.info(f"VPA classifier: {correct}/{len(y)} = {correct/len(y)*100:.1f}% training accuracy")

