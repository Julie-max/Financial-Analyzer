"""
Transaction Spend Classifier — Training Script
Character n-gram TF-IDF + LinearSVC

Key design decisions:
- Character n-grams (2-5 chars) handle UPI-truncated names
- VPA_PERSONAL/VPA_MERCHANT/VPA_QR signals appended during feature extraction
- Heavy "Others" training for P2P personal transfers (the dominant pattern)
- No rule-based logic — pure supervised ML
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Tuple, Dict

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------
CATEGORIES = ["Food", "Travel", "Shopping", "Bills", "Entertainment", "Others"]

# ---------------------------------------------------------------------------
# Labeled training dataset
# ---------------------------------------------------------------------------
TRAINING_DATA: List[Tuple[str, str]] = [
    # ══════════════════════════════════════════════════════════════════════
    # FOOD — restaurants, cafes, food delivery, groceries
    # ══════════════════════════════════════════════════════════════════════
    ("SWIGGY ORDER PAYMENT VPA_MERCHANT", "Food"),
    ("ZOMATO FOOD DELIVERY VPA_MERCHANT", "Food"),
    ("DOMINOS PIZZA ONLINE VPA_MERCHANT", "Food"),
    ("MCDONALDS RESTAURANT", "Food"),
    ("KFC OUTLET PAYMENT", "Food"),
    ("SUBWAY SANDWICHES", "Food"),
    ("PIZZA HUT ORDER", "Food"),
    ("STARBUCKS COFFEE", "Food"),
    ("CAFE COFFEE DAY", "Food"),
    ("BARBEQUE NATION DINING", "Food"),
    ("HOTEL SARAVANA BHAVAN", "Food"),
    ("RESTAURANT BILL PAYMENT", "Food"),
    ("GROCERY STORE PURCHASE", "Food"),
    ("BIG BASKET ONLINE GROCERY VPA_MERCHANT", "Food"),
    ("BLINKIT QUICK COMMERCE VPA_MERCHANT", "Food"),
    ("ZEPTO GROCERY ORDER VPA_MERCHANT", "Food"),
    ("ZEPTO MA VPA_MERCHANT", "Food"),
    ("DUNZO DELIVERY FOOD", "Food"),
    ("UBER EATS ORDER VPA_MERCHANT", "Food"),
    ("DOORDASH FOOD DELIVERY", "Food"),
    ("GRUBHUB ORDER PAYMENT", "Food"),
    ("FOOD PANDA ORDER", "Food"),
    ("FRESHMENU FOOD ORDER", "Food"),
    ("FAASOS FOOD DELIVERY", "Food"),
    ("BEHROUZ BIRYANI ORDER", "Food"),
    ("BURGER KING PAYMENT", "Food"),
    ("NANDOS RESTAURANT", "Food"),
    ("JUST EAT ORDER", "Food"),
    ("DELIVEROO FOOD DELIVERY", "Food"),

    # Real UPI food (small vendors with QR/merchant VPA)
    ("MK FOODS VPA_MERCHANT", "Food"),
    ("AMMAN CO VPA_QR", "Food"),
    ("AMMAN CO VPA_MERCHANT", "Food"),
    ("JOE COFF VPA_MERCHANT", "Food"),
    ("HOTEL GO VPA_QR", "Food"),
    ("NAMMA AM VPA_QR", "Food"),
    ("NAMMA AM VPA_PERSONAL", "Food"),
    ("ZAMCO FO VPA_QR", "Food"),
    ("WINE HIL VPA_QR", "Food"),
    ("CHAI FI VPA_MERCHANT", "Food"),
    ("EACHANARI VPA_QR", "Food"),
    ("LAYALEE VPA_QR", "Food"),
    ("ANNAPOORAN VPA_QR", "Food"),
    ("INIYAAS RE VPA_QR", "Food"),
    ("DIVYANAND LUNCH VPA_PERSONAL", "Food"),
    ("VARA MILAG LUNCH VPA_QR", "Food"),
    ("DEEPA DHAR LUNCH VPA_PERSONAL", "Food"),
    ("AIRPORT SR COFFEE VPA_PERSONAL", "Food"),
    ("THIRD WAVE COFFEE VPA_MERCHANT", "Food"),
    ("CANTEEN CO VPA_MERCHANT", "Food"),
    ("MENAKA VPA_QR", "Food"),
    ("KAY ESS VPA_QR", "Food"),
    ("SRI SARA VPA_MERCHANT", "Food"),
    ("PRS BROL VPA_MERCHANT", "Food"),
    ("BRINDHAC VPA_MERCHANT", "Food"),
    ("MARIYA J VPA_QR", "Food"),
    ("BE KIND VPA_MERCHANT", "Food"),
    ("VETRI FU VPA_QR", "Food"),

    # Food — without VPA (general patterns)
    ("SWIGGY", "Food"),
    ("ZOMATO", "Food"),
    ("HOTEL GOPURAM", "Food"),
    ("RESTAURANT", "Food"),
    ("BAKERY", "Food"),
    ("CAFE", "Food"),
    ("TEA STALL", "Food"),
    ("JUICE BAR", "Food"),
    ("CANTEEN FOOD", "Food"),
    ("TIFFIN SERVICE", "Food"),
    ("MILK DELIVERY", "Food"),

    # ══════════════════════════════════════════════════════════════════════
    # TRAVEL — transport, fuel, hotels, flights, trains
    # ══════════════════════════════════════════════════════════════════════
    ("IRCTC TRAIN TICKET VPA_MERCHANT", "Travel"),
    ("INDIAN R VPA_MERCHANT", "Travel"),
    ("MAKEMYTRIP FLIGHT BOOKING VPA_MERCHANT", "Travel"),
    ("MAKEMYTR VPA_MERCHANT", "Travel"),
    ("GOIBIBO HOTEL BOOKING", "Travel"),
    ("OYO ROOMS BOOKING VPA_MERCHANT", "Travel"),
    ("UBER RIDE PAYMENT VPA_MERCHANT", "Travel"),
    ("OLA CABS RIDE VPA_MERCHANT", "Travel"),
    ("RAPIDO BIKE TAXI VPA_MERCHANT", "Travel"),
    ("INDIGO AIRLINES TICKET", "Travel"),
    ("AIR INDIA FLIGHT BOOKING", "Travel"),
    ("SPICEJET TICKET", "Travel"),
    ("VISTARA AIRLINES", "Travel"),
    ("REDBUS TICKET VPA_MERCHANT", "Travel"),
    ("ABHIBUS TICKET", "Travel"),
    ("METRO CARD RECHARGE", "Travel"),
    ("FASTAG TOLL RECHARGE", "Travel"),
    ("PETROL PUMP FUEL PURCHASE", "Travel"),
    ("HP PETROL STATION", "Travel"),
    ("INDIAN OIL FUEL", "Travel"),
    ("BHARAT PETROLEUM FUEL", "Travel"),
    ("IOCL SHAN VPA_MERCHANT", "Travel"),
    ("PARKING CHARGES", "Travel"),
    ("TOLL PLAZA PAYMENT", "Travel"),
    ("LYFT RIDE PAYMENT", "Travel"),
    ("AIRBNB ACCOMMODATION", "Travel"),
    ("BOOKING COM HOTEL", "Travel"),
    ("EXPEDIA TRAVEL", "Travel"),
    ("EMIRATES AIRLINES", "Travel"),
    ("BRITISH AIRWAYS", "Travel"),
    ("AMTRAK TRAIN TICKET", "Travel"),
    ("CHENNAI METRO VPA_MERCHANT", "Travel"),
    ("PAYTM TRAV VPA_MERCHANT", "Travel"),
    ("CMRL VPA_MERCHANT", "Travel"),

    # Real UPI travel (auto/cab with personal VPA = still travel)
    ("SENTHIL CAB VPA_PERSONAL", "Travel"),
    ("NAGARAJ AUTO VPA_PERSONAL", "Travel"),
    ("KALAI SELV AUTO VPA_PERSONAL", "Travel"),
    ("KARTHIKEYA RIDE VPA_PERSONAL", "Travel"),

    # ══════════════════════════════════════════════════════════════════════
    # SHOPPING — online & offline retail, electronics, clothing
    # ══════════════════════════════════════════════════════════════════════
    ("AMAZON ONLINE PURCHASE VPA_MERCHANT", "Shopping"),
    ("FLIPKART ORDER PAYMENT VPA_MERCHANT", "Shopping"),
    ("MYNTRA FASHION PURCHASE", "Shopping"),
    ("AJIO CLOTHING PURCHASE", "Shopping"),
    ("NYKAA BEAUTY PRODUCTS", "Shopping"),
    ("MEESHO ONLINE SHOPPING", "Shopping"),
    ("RELIANCE DIGITAL PURCHASE", "Shopping"),
    ("CROMA ELECTRONICS STORE", "Shopping"),
    ("APPLE STORE PURCHASE", "Shopping"),
    ("SAMSUNG STORE PAYMENT", "Shopping"),
    ("IKEA FURNITURE PURCHASE", "Shopping"),
    ("ZARA FASHION PURCHASE", "Shopping"),
    ("H M CLOTHING PURCHASE", "Shopping"),
    ("DECATHLON SPORTS GOODS", "Shopping"),
    ("NIKE STORE PURCHASE", "Shopping"),
    ("EBAY ONLINE PURCHASE", "Shopping"),
    ("WALMART RETAIL PURCHASE", "Shopping"),
    ("TARGET STORE PURCHASE", "Shopping"),
    ("BEST BUY ELECTRONICS", "Shopping"),
    ("HOME DEPOT PURCHASE", "Shopping"),
    ("GHARSOAPS VPA_MERCHANT", "Shopping"),

    # ══════════════════════════════════════════════════════════════════════
    # BILLS — utilities, subscriptions, EMIs, insurance, taxes
    # ══════════════════════════════════════════════════════════════════════
    ("ELECTRICITY BILL PAYMENT", "Bills"),
    ("WATER BILL PAYMENT", "Bills"),
    ("GAS BILL PAYMENT", "Bills"),
    ("BROADBAND INTERNET BILL", "Bills"),
    ("MOBILE RECHARGE PAYMENT", "Bills"),
    ("AIRTEL POSTPAID BILL VPA_MERCHANT", "Bills"),
    ("AIRTEL VPA_MERCHANT", "Bills"),
    ("JIO MOBILE BILL VPA_MERCHANT", "Bills"),
    ("JIOFIBER P VPA_MERCHANT", "Bills"),
    ("VODAFONE BILL PAYMENT", "Bills"),
    ("BSNL LANDLINE BILL", "Bills"),
    ("HOUSE RENT PAYMENT", "Bills"),
    ("INSURANCE PREMIUM PAYMENT", "Bills"),
    ("LIC PREMIUM PAYMENT", "Bills"),
    ("HOME LOAN EMI PAYMENT", "Bills"),
    ("PERSONAL LOAN EMI", "Bills"),
    ("CREDIT CARD BILL PAYMENT", "Bills"),
    ("ICICI BANK CREDIT CA", "Bills"),
    ("SBI CARDS VPA_MERCHANT", "Bills"),
    ("MUNICIPAL TAX PAYMENT", "Bills"),
    ("PROPERTY TAX PAYMENT", "Bills"),
    ("INCOME TAX PAYMENT", "Bills"),
    ("SCHOOL FEES PAYMENT", "Bills"),
    ("HOSPITAL BILL PAYMENT", "Bills"),
    ("DOCTOR CONSULTATION FEE", "Bills"),
    ("NETFLIX SUBSCRIPTION VPA_MERCHANT", "Bills"),
    ("SPOTIFY PREMIUM VPA_MERCHANT", "Bills"),
    ("SPOTIFY IN VPA_MERCHANT", "Bills"),
    ("LINKEDIN VPA_MERCHANT", "Bills"),
    ("APPLE MEDI VPA_MERCHANT", "Bills"),
    ("GOOGLE IND VPA_MERCHANT", "Bills"),
    ("GOOGLE I VPA_MERCHANT", "Bills"),
    ("GOOGLE P VPA_MERCHANT", "Bills"),
    ("AWS INDIA VPA_MERCHANT", "Bills"),
    ("EUREKA FOR VPA_MERCHANT", "Bills"),
    ("TANGEDCO VPA_MERCHANT", "Bills"),
    ("ECHALLAN VPA_MERCHANT", "Bills"),
    ("CBDT VPA_MERCHANT", "Bills"),
    ("GYM MEMBERSHIP PAYMENT", "Bills"),
    ("SOCIETY MAINTENANCE CHARGES", "Bills"),

    # ══════════════════════════════════════════════════════════════════════
    # ENTERTAINMENT — movies, gaming, events, streaming
    # ══════════════════════════════════════════════════════════════════════
    ("BOOKMYSHOW MOVIE TICKET VPA_MERCHANT", "Entertainment"),
    ("PVR CINEMAS TICKET", "Entertainment"),
    ("INOX MOVIES TICKET", "Entertainment"),
    ("STEAM GAME PURCHASE", "Entertainment"),
    ("PLAYSTATION STORE PURCHASE", "Entertainment"),
    ("XBOX GAME PASS PAYMENT", "Entertainment"),
    ("CONCERT TICKET BOOKING", "Entertainment"),
    ("LIVE EVENT TICKET", "Entertainment"),
    ("SPORTS EVENT TICKET", "Entertainment"),
    ("THEME PARK ENTRY TICKET", "Entertainment"),
    ("GAMING ZONE PAYMENT", "Entertainment"),
    ("BOWLING ALLEY PAYMENT", "Entertainment"),
    ("BAR DRINKS PAYMENT", "Entertainment"),
    ("PUB PAYMENT", "Entertainment"),
    ("COMEDY SHOW TICKET", "Entertainment"),
    ("MUSEUM TICKET", "Entertainment"),
    ("UDEMY COURSE PURCHASE", "Entertainment"),
    ("COURSERA COURSE PURCHASE", "Entertainment"),

    # Zerodha / investments → Entertainment (per original categories)
    ("ICCL ZEROD VPA_MERCHANT", "Entertainment"),
    ("ZERODHA VPA_MERCHANT", "Entertainment"),
    ("ZERODHA BROKING", "Entertainment"),

    # ══════════════════════════════════════════════════════════════════════
    # OTHERS — P2P transfers, salary, investments, ATM, unknown
    # This is the CRITICAL category: most real-world UPI payments to
    # individuals (personal VPA) should land here.
    # ══════════════════════════════════════════════════════════════════════

    # Salary and income
    ("PENTAFOX TECHNOLOGIES PVT LTD", "Others"),
    ("SAAFE TECHNOLOG", "Others"),
    ("SAGILITY LIMITED", "Others"),
    ("SALARY CREDIT", "Others"),
    ("SALARY FOR JULY", "Others"),
    ("SALARY FOR AUGUST", "Others"),
    ("SALARY FOR SEPTEMBER", "Others"),
    ("SALARY FOR OCTOBER", "Others"),
    ("SALARY FOR NOVEMBER", "Others"),
    ("SALARY FOR DECEMBER", "Others"),

    # Fund transfers and banking
    ("VIGNESH SA", "Others"),
    ("FUND TRANSFER", "Others"),
    ("NEFT TRANSFER", "Others"),
    ("IMPS TRANSFER", "Others"),
    ("RTGS FUND TRANSFER", "Others"),
    ("ATM CASH WITHDRAWAL", "Others"),
    ("CASH DEPOSIT", "Others"),
    ("INTEREST CREDIT", "Others"),
    ("DIVIDEND CREDIT", "Others"),
    ("HOME LOAN KVLPM", "Others"),
    ("HOUSE EXPENSE", "Others"),
    ("FAMILY EXPENSE", "Others"),
    ("FAMILY", "Others"),
    ("DEBT REPAY", "Others"),
    ("EMI PENTA", "Others"),
    ("PL EMI", "Others"),
    ("BALANCE LIGHTS", "Others"),
    ("RVH CAPITAL", "Others"),
    ("RVH INVESTMENT", "Others"),
    ("GROWW INVE", "Others"),
    ("MUTUAL FUND PURCHASE", "Others"),
    ("CASH TRANSFER", "Others"),
    ("CASH WDL", "Others"),

    # P2P transfers with VPA_PERSONAL signal (the key pattern)
    ("MR KALID VPA_PERSONAL", "Others"),
    ("SANTHI S VPA_PERSONAL", "Others"),
    ("C BRIANT VPA_PERSONAL", "Others"),
    ("THIRUMAN VPA_PERSONAL", "Others"),
    ("SATHISH VPA_PERSONAL", "Others"),
    ("R RAKKU VPA_PERSONAL", "Others"),
    ("S VELUCH VPA_PERSONAL", "Others"),
    ("THIRUMAL VPA_PERSONAL", "Others"),
    ("MENAKA VPA_PERSONAL", "Others"),
    ("PARAMESW VPA_PERSONAL", "Others"),
    ("FELLOWKA VPA_PERSONAL", "Others"),
    ("J MELVIN VPA_PERSONAL", "Others"),
    ("B MUHAMM VPA_PERSONAL", "Others"),
    ("RDEVI VPA_PERSONAL", "Others"),
    ("VIVIN RA VPA_PERSONAL", "Others"),
    ("MOHANRAJ VPA_PERSONAL", "Others"),
    ("MR PRAKA VPA_PERSONAL", "Others"),
    ("ESLIN JO VPA_PERSONAL", "Others"),
    ("MR RAMAM VPA_PERSONAL", "Others"),
    ("DINESH R VPA_PERSONAL", "Others"),
    ("POOSAI VPA_PERSONAL", "Others"),
    ("RAVIKUMA VPA_PERSONAL", "Others"),
    ("N KRUBHA VPA_PERSONAL", "Others"),
    ("78454227 VPA_PERSONAL", "Others"),
    ("MR M SOM VPA_PERSONAL", "Others"),
    ("YUVARAJA VPA_PERSONAL", "Others"),
    ("SATHEESH VPA_PERSONAL", "Others"),
    ("RAINBOW VPA_PERSONAL", "Others"),
    ("VEERAPAT VPA_PERSONAL", "Others"),
    ("BALAJI T VPA_MERCHANT", "Others"),
    ("MCHANDRASE VPA_PERSONAL", "Others"),
    ("MRS NEELAV VPA_PERSONAL", "Others"),
    ("D NAVEEN K VPA_PERSONAL", "Others"),
    ("SURESH C VPA_PERSONAL", "Others"),
    ("R PARAMESH VPA_PERSONAL", "Others"),
    ("VIJAYAN RA VPA_PERSONAL", "Others"),
    ("ALEXANDER VPA_PERSONAL", "Others"),
    ("RATNAA SHR VPA_QR", "Others"),
    ("RASOOL MYD VPA_PERSONAL", "Others"),
    ("VALLAVARAJ VPA_PERSONAL", "Others"),
    ("RAMYA GOWR VPA_PERSONAL", "Others"),
    ("CHANDRASEK VPA_PERSONAL", "Others"),
    ("LOGESHWHAR VPA_PERSONAL", "Others"),
    ("SHANKAR B VPA_PERSONAL", "Others"),
    ("MANIKANDAN VPA_PERSONAL", "Others"),
    ("KARTHIK P VPA_PERSONAL", "Others"),
    ("SASIKUMAR VPA_PERSONAL", "Others"),
    ("MR THIRUMA VPA_PERSONAL", "Others"),
    ("LATHA M VPA_PERSONAL", "Others"),
    ("MADHA ELEC VPA_PERSONAL", "Others"),
    ("GAYATHRI D VPA_PERSONAL", "Others"),
    ("SRIVIDYA S VPA_PERSONAL", "Others"),
    ("SAKTHIVEL VPA_PERSONAL", "Others"),
    ("GOKUL SP VPA_PERSONAL", "Others"),
    ("MR SUGAN S VPA_PERSONAL", "Others"),
    ("SRINAGAV VPA_PERSONAL", "Others"),

    # P2P without VPA signal (still should be Others based on name patterns)
    ("MR KALID", "Others"),
    ("SANTHI S", "Others"),
    ("C BRIANT", "Others"),
    ("THIRUMAN", "Others"),
    ("SATHISH", "Others"),
    ("R RAKKU", "Others"),
    ("S VELUCH", "Others"),
    ("THIRUMAL", "Others"),
    ("PARAMESW", "Others"),
    ("J MELVIN", "Others"),
    ("B MUHAMM", "Others"),
    ("VIVIN RA", "Others"),
    ("MR PRAKA", "Others"),
    ("ESLIN JO", "Others"),
    ("MR RAMAM", "Others"),
    ("DINESH R", "Others"),
    ("POOSAI", "Others"),
    ("RAVIKUMA", "Others"),
    ("N KRUBHA", "Others"),
    ("MR M SOM", "Others"),
    ("YUVARAJA", "Others"),
    ("SATHEESH", "Others"),
    ("RAINBOW", "Others"),
    ("VEERAPAT", "Others"),
    ("FELLOWKA", "Others"),

    # UPI reversals, cashback
    ("UPI REVERSAL", "Others"),
    ("NPCI BHIM VPA_MERCHANT", "Others"),
    ("BHIMCASHBACK VPA_MERCHANT", "Others"),
    ("REFUND CREDIT", "Others"),
    ("CASHBACK CREDIT", "Others"),

    # Google Play refund (credit = Others)
    ("GOOGLE P VPA_MERCHANT", "Others"),

    # ══════════════════════════════════════════════════════════════════════
    # Additional VPA_PERSONAL patterns — ensures the "personal" signal
    # STRONGLY pushes towards Others regardless of payee name
    # ══════════════════════════════════════════════════════════════════════
    ("UNKNOWN NAME VPA_PERSONAL", "Others"),
    ("PERSON PAYMENT VPA_PERSONAL", "Others"),
    ("TRANSFER VPA_PERSONAL", "Others"),
    ("UPI PAYMENT VPA_PERSONAL", "Others"),
    ("SENT MONEY VPA_PERSONAL", "Others"),
    ("RECEIVED VPA_PERSONAL", "Others"),
    ("PAID VPA_PERSONAL", "Others"),
    ("KUMAR VPA_PERSONAL", "Others"),
    ("SINGH VPA_PERSONAL", "Others"),
    ("SHARMA VPA_PERSONAL", "Others"),
    ("PATEL VPA_PERSONAL", "Others"),
    ("RAVI VPA_PERSONAL", "Others"),
    ("SURESH VPA_PERSONAL", "Others"),
    ("RAMESH VPA_PERSONAL", "Others"),
    ("MOHAMMED VPA_PERSONAL", "Others"),
    ("ARUN VPA_PERSONAL", "Others"),
    ("PRIYA VPA_PERSONAL", "Others"),
    ("LAKSHMI VPA_PERSONAL", "Others"),
    ("VENKAT VPA_PERSONAL", "Others"),
    ("ANAND VPA_PERSONAL", "Others"),
    ("GANESH VPA_PERSONAL", "Others"),
    ("NAVEEN VPA_PERSONAL", "Others"),
    ("PRAKASH VPA_PERSONAL", "Others"),
    ("RAJA VPA_PERSONAL", "Others"),
    ("SELVAM VPA_PERSONAL", "Others"),
    ("MURUGAN VPA_PERSONAL", "Others"),
    ("SENTHIL VPA_PERSONAL", "Others"),
    ("BALA VPA_PERSONAL", "Others"),
    ("GOPI VPA_PERSONAL", "Others"),
    ("MANI VPA_PERSONAL", "Others"),
    ("DEVI VPA_PERSONAL", "Others"),
]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Normalise a transaction description for ML feature extraction.
    - Lowercase
    - Remove special characters (keep alphanumeric + spaces + underscore for VPA signals)
    - Collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_pipeline(model_type: str = "svm") -> Pipeline:
    """
    Build a scikit-learn Pipeline with character n-gram TF-IDF + LinearSVC.
    """
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        max_features=50_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    if model_type == "logreg":
        clf = LogisticRegression(
            C=5.0,
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
            class_weight="balanced",
        )
    else:
        clf = LinearSVC(
            C=1.0,
            dual="auto",
            max_iter=2000,
            class_weight="balanced",
        )

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def train(
    data: List[Tuple[str, str]] = None,
    model_type: str = "svm",
    model_dir: str = None,
    cv_folds: int = 5,
) -> Dict:
    """
    Train the classifier and save the model.
    """
    if data is None:
        data = TRAINING_DATA

    if model_dir is None:
        model_dir = Path(__file__).parent.parent / "models"
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    descriptions, labels = zip(*data)
    descriptions = [preprocess_text(d) for d in descriptions]

    logger.info(f"Training on {len(descriptions)} samples | model={model_type}")
    logger.info(f"Categories: {sorted(set(labels))}")

    pipeline = build_pipeline(model_type)

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, descriptions, labels, cv=cv, scoring="f1_macro")
    logger.info(
        f"Cross-validation F1 (macro): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
    )

    # Final fit on all data
    pipeline.fit(descriptions, labels)

    # In-sample metrics
    preds = pipeline.predict(descriptions)
    report = classification_report(labels, preds, target_names=sorted(set(labels)))
    logger.info(f"\nClassification Report (train set):\n{report}")

    # Save artifacts
    model_path = model_dir / "classifier_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save label list
    label_path = model_dir / "categories.json"
    with open(label_path, "w") as f:
        json.dump(CATEGORIES, f)
    logger.info(f"Categories saved to {label_path}")

    return {
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "model_path": str(model_path),
        "n_samples": len(descriptions),
    }


if __name__ == "__main__":
    metrics = train()
    print(f"\nTraining complete. CV F1: {metrics['cv_f1_mean']:.3f}")
