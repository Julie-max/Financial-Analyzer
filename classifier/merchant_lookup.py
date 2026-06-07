"""
Merchant Lookup — Stage 1 of the two-stage classifier.

Maps known UPI VPA fragments, truncated names, and merchant codes
to their real name and category.

This is DATA-driven, not rule-driven:
- The lookup table is a JSON-like dict that can be extended without code changes
- Matching uses substring search + fuzzy matching (RapidFuzz)
- No hardcoded if/else logic per merchant

Why this exists:
  UPI truncates payee names to 8 chars. "BOOKMYSH" has zero character overlap
  with "ENTERTAINMENT" in TF-IDF space. A lookup resolves the known ones;
  the ML classifier handles everything else.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Merchant database
# Format: "fragment_to_match": ("Clean Display Name", "Category")
# Matching: case-insensitive substring search on the cleaned description
# ---------------------------------------------------------------------------
MERCHANT_DB = {
    # ── Entertainment ──────────────────────────────────────────────────────
    "bookmysh":     ("BookMyShow", "Entertainment"),
    "bookmyshow":   ("BookMyShow", "Entertainment"),
    "pvr":          ("PVR Cinemas", "Entertainment"),
    "inox":         ("INOX Movies", "Entertainment"),
    "cinepolis":    ("Cinepolis", "Entertainment"),
    "steam":        ("Steam", "Entertainment"),
    "playstation":  ("PlayStation", "Entertainment"),
    "xbox":         ("Xbox", "Entertainment"),
    "zerodha":      ("Zerodha", "Entertainment"),
    "iccl zerod":   ("Zerodha", "Entertainment"),
    "zerodhamf":    ("Zerodha", "Entertainment"),
    "zerodhafund":  ("Zerodha", "Entertainment"),
    "zerodha brok": ("Zerodha", "Entertainment"),
    "dream11":      ("Dream11", "Entertainment"),
    "wynk":         ("Wynk Music", "Entertainment"),
    "gaana":        ("Gaana", "Entertainment"),
    "jiosaavn":     ("JioSaavn", "Entertainment"),
    "bigtreeent":   ("BookMyShow", "Entertainment"),
    "bigtree":      ("BookMyShow", "Entertainment"),
    "big tree":     ("BookMyShow", "Entertainment"),
    "autope p":     ("IRCTC AutoPay", "Travel"),
    "irctcautop":   ("IRCTC AutoPay", "Travel"),

    # ── Bills (subscriptions that are recurring) ──────────────────────────
    "netflix":      ("Netflix", "Bills"),
    "hotstar":      ("Hotstar", "Bills"),
    "disneyplus":   ("Disney+Hotstar", "Bills"),
    "spotify":      ("Spotify", "Bills"),
    "spotifyindi":  ("Spotify", "Bills"),
    "spotifyindia": ("Spotify", "Bills"),
    "apple music":  ("Apple Music", "Bills"),
    "youtube":      ("YouTube Premium", "Bills"),
    "amazon prime": ("Amazon Prime", "Bills"),
    "prime video":  ("Prime Video", "Bills"),

    # ── Travel ─────────────────────────────────────────────────────────────
    "makemytr":     ("MakeMyTrip", "Travel"),
    "makemytrip":   ("MakeMyTrip", "Travel"),
    "make my tr":   ("MakeMyTrip", "Travel"),
    "irctc":        ("IRCTC", "Travel"),
    "indianrail":   ("Indian Railways", "Travel"),
    "indian r":     ("Indian Railways", "Travel"),
    "indian rai":   ("Indian Railways", "Travel"),
    "redbus":       ("RedBus", "Travel"),
    "redbus1":      ("RedBus", "Travel"),
    "redbus i":     ("RedBus", "Travel"),
    "goibibo":      ("Goibibo", "Travel"),
    "oyo":          ("OYO Rooms", "Travel"),
    "oyorooms":     ("OYO Rooms", "Travel"),
    "uber":         ("Uber", "Travel"),
    "ola":          ("Ola Cabs", "Travel"),
    "rapido":       ("Rapido", "Travel"),
    "indigo":       ("IndiGo Airlines", "Travel"),
    "air india":    ("Air India", "Travel"),
    "airindia":     ("Air India", "Travel"),
    "spicejet":     ("SpiceJet", "Travel"),
    "vistara":      ("Vistara", "Travel"),
    "paytm trav":   ("Paytm Travel", "Travel"),
    "travel1paytm": ("Paytm Travel", "Travel"),
    "iocl":         ("IOCL Fuel", "Travel"),
    "petrol":       ("Petrol", "Travel"),
    "fastag":       ("FASTag", "Travel"),
    "cmrl":         ("Chennai Metro", "Travel"),
    "chennai me":   ("Chennai Metro", "Travel"),
    "metro":        ("Metro Rail", "Travel"),
    "expedia":      ("Expedia", "Travel"),
    "agoda":        ("Agoda", "Travel"),
    "abhibus":      ("AbhiBus", "Travel"),
    "town house":   ("Town House Hotel", "Travel"),
    "bhimanna h cab": ("Bhimanna Cab", "Travel"),
    "cab ride":     ("Cab Ride", "Travel"),
    "auto ride":    ("Auto Ride", "Travel"),
    "shanmugana auto": ("Auto Ride", "Travel"),
    "kamaraj aka":  ("Kamaraj Transport", "Travel"),

    # ── Food ───────────────────────────────────────────────────────────────
    "swiggy":       ("Swiggy", "Food"),
    "swiggystor":   ("Swiggy", "Food"),
    "zomato":       ("Zomato", "Food"),
    "payzomato":    ("Zomato", "Food"),
    "zomatolim":    ("Zomato", "Food"),
    "dominos":      ("Dominos", "Food"),
    "pizza hut":    ("Pizza Hut", "Food"),
    "kfc":          ("KFC", "Food"),
    "mcdonalds":    ("McDonald's", "Food"),
    "burger king":  ("Burger King", "Food"),
    "starbucks":    ("Starbucks", "Food"),
    "cafe coffee":  ("Cafe Coffee Day", "Food"),
    "joe coff":     ("Joe Coffee", "Food"),
    "chai fi":      ("Chai Fi", "Food"),
    "chaifi":       ("Chai Fi", "Food"),
    "zepto":        ("Zepto", "Food"),
    "zeptoonlin":   ("Zepto", "Food"),
    "zeptomarke":   ("Zepto", "Food"),
    "blinkit":      ("Blinkit", "Food"),
    "bigbasket":    ("BigBasket", "Food"),
    "big basket":   ("BigBasket", "Food"),
    "grofers":      ("Grofers", "Food"),
    "dunzo":        ("Dunzo", "Food"),
    "instamart":    ("Swiggy Instamart", "Food"),
    "ubereats":     ("Uber Eats", "Food"),
    "faasos":       ("Faasos", "Food"),
    "behrouz":      ("Behrouz Biryani", "Food"),
    "freshmenu":    ("FreshMenu", "Food"),
    "wine hil":     ("Wine Hills", "Food"),
    "wine hill":    ("Wine Hills", "Food"),
    "gopuram":      ("Gopuram", "Food"),
    "hotel go":     ("Hotel Gopuram", "Food"),
    "eachanari":    ("Eachanari", "Food"),
    "amman co":     ("Amman Co", "Food"),
    "namma am":     ("Namma Amman", "Food"),
    "zamco fo":     ("Zamco Food", "Food"),
    "thats y fo":   ("That's Y Food", "Food"),
    "layalee":      ("Layalee", "Food"),
    "brindhac":     ("Brindha Cafe", "Food"),
    "mk foods":     ("MK Foods", "Food"),
    "annapooran":   ("Annapoorna", "Food"),
    "iniyaas re":   ("Iniyaas Restaurant", "Food"),
    "third wave":   ("Third Wave Coffee", "Food"),
    "thirdwavecoffe": ("Third Wave Coffee", "Food"),
    "prs brol":     ("PRS Broilers", "Food"),
    "sri sara":     ("Sri Saravana", "Food"),
    "be kind":      ("Be Kind Cafe", "Food"),
    "kay ess":      ("Kay Ess", "Food"),
    "laxmi hote":   ("Laxmi Hotel", "Food"),
    "le grace":     ("Le Grace Restaurant", "Food"),
    "mr vegetab":   ("Vegetables", "Food"),
    "basil fnb":    ("Basil FnB", "Food"),
    "gopuram":      ("Gopuram", "Food"),

    # ── Bills ──────────────────────────────────────────────────────────────
    "jiofiber":     ("JioFiber", "Bills"),
    "jiofiberpre":  ("JioFiber", "Bills"),
    "jio postpa":   ("Jio Postpaid", "Bills"),
    "jio mobile":   ("Jio", "Bills"),
    "airtel":       ("Airtel", "Bills"),
    "vodafone":     ("Vodafone", "Bills"),
    "bsnl":         ("BSNL", "Bills"),
    "electricity":  ("Electricity Bill", "Bills"),
    "tangedco":     ("TANGEDCO", "Bills"),
    "bescom":       ("BESCOM", "Bills"),
    "linkedin":     ("LinkedIn", "Bills"),
    "apple me":     ("Apple", "Bills"),
    "appleservi":   ("Apple", "Bills"),
    "apple medi":   ("Apple", "Bills"),
    "google ind":   ("Google", "Bills"),
    "google i":     ("Google", "Bills"),
    "gpayrechar":   ("Google Recharge", "Bills"),
    "gpayutili":    ("Google Utility", "Bills"),
    "aws india":    ("AWS", "Bills"),
    "amazonaws":    ("AWS", "Bills"),
    "icici bank credit": ("ICICI Credit Card", "Bills"),
    "sbi cards":    ("SBI Credit Card", "Bills"),
    "personal loan": ("Personal Loan EMI", "Bills"),
    "home loan":    ("Home Loan EMI", "Bills"),
    "lic":          ("LIC Premium", "Bills"),
    "insurance":    ("Insurance", "Bills"),
    "cbdt":         ("Income Tax", "Bills"),
    "echallan":     ("E-Challan", "Bills"),
    "atria co":     ("Atria Convergence", "Bills"),
    "eureka for":   ("Eureka Forbes", "Bills"),
    "eurekaforb":   ("Eureka Forbes", "Bills"),

    # ── Shopping ───────────────────────────────────────────────────────────
    "amazon":       ("Amazon", "Shopping"),
    "flipkart":     ("Flipkart", "Shopping"),
    "myntra":       ("Myntra", "Shopping"),
    "ajio":         ("AJIO", "Shopping"),
    "nykaa":        ("Nykaa", "Shopping"),
    "meesho":       ("Meesho", "Shopping"),
    "snapdeal":     ("Snapdeal", "Shopping"),
    "croma":        ("Croma", "Shopping"),
    "reliance digital": ("Reliance Digital", "Shopping"),
    "decathlon":    ("Decathlon", "Shopping"),
    "zudio":        ("Zudio", "Shopping"),
    "zudio ba":     ("Zudio", "Shopping"),
    "bhima":        ("Bhima Jewellery", "Shopping"),
    "jewel":        ("Jewellery", "Shopping"),
    "ekart":        ("Ekart Logistics", "Shopping"),
    "ruptub":       ("Treebo", "Shopping"),

    # ── Others ─────────────────────────────────────────────────────────────
    "salary":       ("Salary", "Others"),
    "hevo techno":  ("Hevo Technologies", "Others"),
    "hevo technolo": ("Hevo Technologies", "Others"),
    "pentafox":     ("Pentafox Technologies", "Others"),
    "saafe techno": ("SAAFE Technologies", "Others"),
    "sagility":     ("Sagility Limited", "Others"),
    "groww":        ("Groww", "Others"),
    "groww inve":   ("Groww", "Others"),
    "neft":         ("NEFT Transfer", "Others"),
    "imps":         ("IMPS Transfer", "Others"),
    "interest credit": ("Interest Credit", "Others"),
    "upi reversal": ("UPI Reversal", "Others"),
    "bhimcashback": ("BHIM Cashback", "Others"),
    "npci bhim":    ("BHIM Cashback", "Others"),
    "fund transfer": ("Fund Transfer", "Others"),
    "house expense": ("House Expense", "Others"),
    "family":       ("Family Transfer", "Others"),
    "debt repay":   ("Debt Repayment", "Others"),
    "cash wdl":     ("Cash Withdrawal", "Others"),
    "atm withdrawal": ("ATM Withdrawal", "Others"),
    "atm wdl":      ("ATM Withdrawal", "Others"),
    "to wife":      ("Family Transfer", "Others"),
    "to amma":      ("Family Transfer", "Others"),
    "to ramya":     ("Family Transfer", "Others"),
    "emergency":    ("Emergency Transfer", "Others"),
    "vignesh sa":   ("Self Transfer", "Others"),
}


class MerchantLookup:
    """
    Stage 1 classifier: looks up known merchants by substring matching.
    Returns (clean_name, category) or None if not found.
    
    Checks both the built-in MERCHANT_DB and user-added merchants from feedback.
    """

    def __init__(self):
        # Pre-process keys to lowercase for fast lookup
        self._db = {k.lower(): v for k, v in MERCHANT_DB.items()}
        
        # Load user-added merchants and merge (user overrides built-in)
        try:
            from classifier.feedback import load_user_merchants
            user_merchants = load_user_merchants()
            self._db.update(user_merchants)
        except Exception:
            pass
        
        # Sort by length descending so longer (more specific) matches win
        self._keys = sorted(self._db.keys(), key=len, reverse=True)

    def lookup(self, description: str) -> Optional[Tuple[str, str]]:
        """
        Try to match description against known merchants.

        Returns:
            (clean_name, category) tuple if found, None otherwise.
        """
        desc_lower = description.lower().strip()

        # Exact match first
        if desc_lower in self._db:
            return self._db[desc_lower]

        # Substring match — longer keys checked first (more specific wins)
        for key in self._keys:
            if key in desc_lower:
                return self._db[key]

        return None
    
    def reload(self):
        """Reload the merchant database (call after adding new merchants)."""
        self.__init__()
