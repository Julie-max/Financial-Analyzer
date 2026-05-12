"""
CRF-Based Transaction Description Parser

Uses Conditional Random Fields (CRF) — the classic pre-deep-learning NLP
algorithm for sequence labeling — to extract merchant names from raw bank
transaction descriptions.

Why CRF:
- Sequence labeling: labels each token in context, not in isolation
- Generalizes across bank formats: learns structural patterns, not bank-specific rules
- No neural network, no GPU, fast CPU inference (~1ms per description)
- Handles UPI, NEFT, IMPS, BIL, ATM, POS formats without format-specific code

Token Labels:
  PREFIX   - transaction type prefix (UPI, NEFT, WDL TFR, BY ONL, etc.)
  REF_NUM  - reference/transaction number (pure digits or alphanumeric codes)
  PAYEE    - merchant/payee name (what we want to extract)
  BANK     - bank code (YESB, HDFC, SBIN, UTIB, etc.)
  VPA      - UPI Virtual Payment Address (merchant@bank format)
  REMARK   - transaction remark/purpose
  SUFFIX   - trailing branch/terminal info (AT 00869 MADURAI, U::00116)
  SEP      - separator (/, -, :, ::)
  OTHER    - anything else

Training data is built from real bank statement descriptions across
CUB, SBI, ICICI, HDFC formats.
"""

import logging
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import joblib

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "crf_parser.joblib"

# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------
LABELS = ["PREFIX", "REF_NUM", "PAYEE", "BANK", "VPA", "REMARK", "SUFFIX", "SEP", "OTHER"]

# ---------------------------------------------------------------------------
# Training data
# Each entry: (raw_description, [(token, label), ...])
# ---------------------------------------------------------------------------
TRAINING_DATA = [
    # ── CUB UPI/DR format ─────────────────────────────────────────────────
    (
        "UPI/DR/116493305468/SUJIT BR/YESB/PAYTMQR6X5/U::00116",
        [("UPI", "PREFIX"), ("/", "SEP"), ("DR", "PREFIX"), ("/", "SEP"),
         ("116493305468", "REF_NUM"), ("/", "SEP"),
         ("SUJIT", "PAYEE"), ("BR", "PAYEE"), ("/", "SEP"),
         ("YESB", "BANK"), ("/", "SEP"), ("PAYTMQR6X5", "VPA"), ("/", "SEP"),
         ("U::00116", "SUFFIX")]
    ),
    (
        "UPI/DR/116503729579/WINE HIL/YESB/Q928397193/U::00116",
        [("UPI", "PREFIX"), ("/", "SEP"), ("DR", "PREFIX"), ("/", "SEP"),
         ("116503729579", "REF_NUM"), ("/", "SEP"),
         ("WINE", "PAYEE"), ("HIL", "PAYEE"), ("/", "SEP"),
         ("YESB", "BANK"), ("/", "SEP"), ("Q928397193", "VPA"), ("/", "SEP"),
         ("U::00116", "SUFFIX")]
    ),
    (
        "UPI/DR/116934097792/BOOKMYSH/YESB/PAYTM-8726/B::00116",
        [("UPI", "PREFIX"), ("/", "SEP"), ("DR", "PREFIX"), ("/", "SEP"),
         ("116934097792", "REF_NUM"), ("/", "SEP"),
         ("BOOKMYSH", "PAYEE"), ("/", "SEP"),
         ("YESB", "BANK"), ("/", "SEP"), ("PAYTM-8726", "VPA"), ("/", "SEP"),
         ("B::00116", "SUFFIX")]
    ),
    (
        "UPI/DR/117658329033/MAKEMYTR/HDFC/MAKEMYTRIP/U::00116",
        [("UPI", "PREFIX"), ("/", "SEP"), ("DR", "PREFIX"), ("/", "SEP"),
         ("117658329033", "REF_NUM"), ("/", "SEP"),
         ("MAKEMYTR", "PAYEE"), ("/", "SEP"),
         ("HDFC", "BANK"), ("/", "SEP"), ("MAKEMYTRIP", "VPA"), ("/", "SEP"),
         ("U::00116", "SUFFIX")]
    ),
    (
        "UPI/CR/116561428498/JEYANTHI/IOBA/JEYANTHIM2/U::00032",
        [("UPI", "PREFIX"), ("/", "SEP"), ("CR", "PREFIX"), ("/", "SEP"),
         ("116561428498", "REF_NUM"), ("/", "SEP"),
         ("JEYANTHI", "PAYEE"), ("/", "SEP"),
         ("IOBA", "BANK"), ("/", "SEP"), ("JEYANTHIM2", "VPA"), ("/", "SEP"),
         ("U::00032", "SUFFIX")]
    ),
    (
        "UPI/DR/607810202565/ZOMATO/HDFC/PAYZOMATO@/UPI::00116",
        [("UPI", "PREFIX"), ("/", "SEP"), ("DR", "PREFIX"), ("/", "SEP"),
         ("607810202565", "REF_NUM"), ("/", "SEP"),
         ("ZOMATO", "PAYEE"), ("/", "SEP"),
         ("HDFC", "BANK"), ("/", "SEP"), ("PAYZOMATO@", "VPA"), ("/", "SEP"),
         ("UPI::00116", "SUFFIX")]
    ),
    (
        "UPI/DR/604032991953/CHAI FI/FDRL/CHAIFI25@F/UP::00116",
        [("UPI", "PREFIX"), ("/", "SEP"), ("DR", "PREFIX"), ("/", "SEP"),
         ("604032991953", "REF_NUM"), ("/", "SEP"),
         ("CHAI", "PAYEE"), ("FI", "PAYEE"), ("/", "SEP"),
         ("FDRL", "BANK"), ("/", "SEP"), ("CHAIFI25@F", "VPA"), ("/", "SEP"),
         ("UP::00116", "SUFFIX")]
    ),
    (
        "UPI/DR/641705915492/REDBUS I/UTIB/REDBUS1ONL/U::00116",
        [("UPI", "PREFIX"), ("/", "SEP"), ("DR", "PREFIX"), ("/", "SEP"),
         ("641705915492", "REF_NUM"), ("/", "SEP"),
         ("REDBUS", "PAYEE"), ("I", "PAYEE"), ("/", "SEP"),
         ("UTIB", "BANK"), ("/", "SEP"), ("REDBUS1ONL", "VPA"), ("/", "SEP"),
         ("U::00116", "SUFFIX")]
    ),
    (
        "UPI/DR/116750249493/KALYANA /YESB/PAYTMQR6UJ/U::00116",
        [("UPI", "PREFIX"), ("/", "SEP"), ("DR", "PREFIX"), ("/", "SEP"),
         ("116750249493", "REF_NUM"), ("/", "SEP"),
         ("KALYANA", "PAYEE"), ("/", "SEP"),
         ("YESB", "BANK"), ("/", "SEP"), ("PAYTMQR6UJ", "VPA"), ("/", "SEP"),
         ("U::00116", "SUFFIX")]
    ),
    # ── SBI UPI/DR format (with WDL TFR prefix and AT suffix) ─────────────
    (
        "WDL TFR UPI/DR/645779753195/HOTEL GO/YESB/paytmqrwlg/UPI 0097692162094 AT 00869 MADURAI",
        [("WDL", "PREFIX"), ("TFR", "PREFIX"),
         ("UPI", "PREFIX"), ("/", "SEP"), ("DR", "PREFIX"), ("/", "SEP"),
         ("645779753195", "REF_NUM"), ("/", "SEP"),
         ("HOTEL", "PAYEE"), ("GO", "PAYEE"), ("/", "SEP"),
         ("YESB", "BANK"), ("/", "SEP"), ("paytmqrwlg", "VPA"), ("/", "SEP"),
         ("UPI", "SUFFIX"), ("0097692162094", "SUFFIX"), ("AT", "SUFFIX"),
         ("00869", "SUFFIX"), ("MADURAI", "SUFFIX")]
    ),
    (
        "WDL TFR UPI/DR/645750525301/AMMAN CO/HDFC/vyapar.175/UPI 0097692162094 AT 00869 MADURAI",
        [("WDL", "PREFIX"), ("TFR", "PREFIX"),
         ("UPI", "PREFIX"), ("/", "SEP"), ("DR", "PREFIX"), ("/", "SEP"),
         ("645750525301", "REF_NUM"), ("/", "SEP"),
         ("AMMAN", "PAYEE"), ("CO", "PAYEE"), ("/", "SEP"),
         ("HDFC", "BANK"), ("/", "SEP"), ("vyapar.175", "VPA"), ("/", "SEP"),
         ("UPI", "SUFFIX"), ("0097692162094", "SUFFIX"), ("AT", "SUFFIX"),
         ("00869", "SUFFIX"), ("MADURAI", "SUFFIX")]
    ),
    (
        "DEP TFR UPI/CR/646042736543/C BRIANT/UBIN/julieschan/UPI 0097737162096 AT 00869 MADURAI",
        [("DEP", "PREFIX"), ("TFR", "PREFIX"),
         ("UPI", "PREFIX"), ("/", "SEP"), ("CR", "PREFIX"), ("/", "SEP"),
         ("646042736543", "REF_NUM"), ("/", "SEP"),
         ("C", "PAYEE"), ("BRIANT", "PAYEE"), ("/", "SEP"),
         ("UBIN", "BANK"), ("/", "SEP"), ("julieschan", "VPA"), ("/", "SEP"),
         ("UPI", "SUFFIX"), ("0097737162096", "SUFFIX"), ("AT", "SUFFIX"),
         ("00869", "SUFFIX"), ("MADURAI", "SUFFIX")]
    ),
    (
        "WDL TFR UPI/DR/646504239229/INDIAN R/UTIB/indianrail/UPI 0097693162093 AT 00869 MADURAI",
        [("WDL", "PREFIX"), ("TFR", "PREFIX"),
         ("UPI", "PREFIX"), ("/", "SEP"), ("DR", "PREFIX"), ("/", "SEP"),
         ("646504239229", "REF_NUM"), ("/", "SEP"),
         ("INDIAN", "PAYEE"), ("R", "PAYEE"), ("/", "SEP"),
         ("UTIB", "BANK"), ("/", "SEP"), ("indianrail", "VPA"), ("/", "SEP"),
         ("UPI", "SUFFIX"), ("0097693162093", "SUFFIX"), ("AT", "SUFFIX"),
         ("00869", "SUFFIX"), ("MADURAI", "SUFFIX")]
    ),
    # ── ICICI UPI format (standard: UPI/PAYEE/VPA/REMARK/BANK/REF) ────────
    (
        "UPI/JIOFiber P/jiofiberprepai/OidS100011/YES BANK L/555308030667/PAYTM507068040",
        [("UPI", "PREFIX"), ("/", "SEP"),
         ("JIOFiber", "PAYEE"), ("P", "PAYEE"), ("/", "SEP"),
         ("jiofiberprepai", "VPA"), ("/", "SEP"),
         ("OidS100011", "REMARK"), ("/", "SEP"),
         ("YES", "BANK"), ("BANK", "BANK"), ("L", "BANK"), ("/", "SEP"),
         ("555308030667", "REF_NUM"), ("/", "SEP"),
         ("PAYTM507068040", "SUFFIX")]
    ),
    (
        "UPI/Swiggy/swiggystores@i/NO REMARKS/ICICI Bank/141705890649/UPIb5fa8324922c5cf",
        [("UPI", "PREFIX"), ("/", "SEP"),
         ("Swiggy", "PAYEE"), ("/", "SEP"),
         ("swiggystores@i", "VPA"), ("/", "SEP"),
         ("NO", "REMARK"), ("REMARKS", "REMARK"), ("/", "SEP"),
         ("ICICI", "BANK"), ("Bank", "BANK"), ("/", "SEP"),
         ("141705890649", "REF_NUM"), ("/", "SEP"),
         ("UPIb5fa8324922c5cf", "SUFFIX")]
    ),
    (
        "UPI/MK FOODS/gpay112517363/Food/AXIS BANK/667650162742/IBL7d50e6ec5dc14e709fc2",
        [("UPI", "PREFIX"), ("/", "SEP"),
         ("MK", "PAYEE"), ("FOODS", "PAYEE"), ("/", "SEP"),
         ("gpay112517363", "VPA"), ("/", "SEP"),
         ("Food", "REMARK"), ("/", "SEP"),
         ("AXIS", "BANK"), ("BANK", "BANK"), ("/", "SEP"),
         ("667650162742", "REF_NUM"), ("/", "SEP"),
         ("IBL7d50e6ec5dc14e709fc2", "SUFFIX")]
    ),
    (
        "UPI/LinkedIn/linkedin.bdsi@/MandateReq/ICICI Bank/561630755744/ICI9345f6059ed24",
        [("UPI", "PREFIX"), ("/", "SEP"),
         ("LinkedIn", "PAYEE"), ("/", "SEP"),
         ("linkedin.bdsi@", "VPA"), ("/", "SEP"),
         ("MandateReq", "REMARK"), ("/", "SEP"),
         ("ICICI", "BANK"), ("Bank", "BANK"), ("/", "SEP"),
         ("561630755744", "REF_NUM"), ("/", "SEP"),
         ("ICI9345f6059ed24", "SUFFIX")]
    ),
    (
        "UPI/ICCL ZEROD/zerodhamf@hdfc/UDpZtZ6kIW/HDFC BANK/100984507168/HDF00123D670BEA",
        [("UPI", "PREFIX"), ("/", "SEP"),
         ("ICCL", "PAYEE"), ("ZEROD", "PAYEE"), ("/", "SEP"),
         ("zerodhamf@hdfc", "VPA"), ("/", "SEP"),
         ("UDpZtZ6kIW", "REMARK"), ("/", "SEP"),
         ("HDFC", "BANK"), ("BANK", "BANK"), ("/", "SEP"),
         ("100984507168", "REF_NUM"), ("/", "SEP"),
         ("HDF00123D670BEA", "SUFFIX")]
    ),
    # ── NEFT format ────────────────────────────────────────────────────────
    (
        "NEFT-INDBN52025073104653733-PENTAFOX TECHNOLOGIES PVT LTD-SALARY FOR JULY-00993",
        [("NEFT", "PREFIX"), ("-", "SEP"),
         ("INDBN52025073104653733", "REF_NUM"), ("-", "SEP"),
         ("PENTAFOX", "PAYEE"), ("TECHNOLOGIES", "PAYEE"), ("PVT", "PAYEE"), ("LTD", "PAYEE"),
         ("-", "SEP"),
         ("SALARY", "REMARK"), ("FOR", "REMARK"), ("JULY", "REMARK"),
         ("-", "SEP"), ("00993", "SUFFIX")]
    ),
    (
        "NEFT-INDBN52025073104765344-PENTAFOX TECHNOLOGIES PVT LTD--259840411735-INDB000",
        [("NEFT", "PREFIX"), ("-", "SEP"),
         ("INDBN52025073104765344", "REF_NUM"), ("-", "SEP"),
         ("PENTAFOX", "PAYEE"), ("TECHNOLOGIES", "PAYEE"), ("PVT", "PAYEE"), ("LTD", "PAYEE"),
         ("-", "SEP"), ("-", "SEP"),
         ("259840411735", "REF_NUM"), ("-", "SEP"), ("INDB000", "SUFFIX")]
    ),
    # ── BIL format ─────────────────────────────────────────────────────────
    (
        "BIL/001046588632/ICICI BANK CREDIT CA/437551711830",
        [("BIL", "PREFIX"), ("/", "SEP"),
         ("001046588632", "REF_NUM"), ("/", "SEP"),
         ("ICICI", "PAYEE"), ("BANK", "PAYEE"), ("CREDIT", "PAYEE"), ("CA", "PAYEE"),
         ("/", "SEP"), ("437551711830", "SUFFIX")]
    ),
    (
        "BIL/INFT/EGZ1180606/House expense/",
        [("BIL", "PREFIX"), ("/", "SEP"),
         ("INFT", "PREFIX"), ("/", "SEP"),
         ("EGZ1180606", "REF_NUM"), ("/", "SEP"),
         ("House", "REMARK"), ("expense", "REMARK"), ("/", "SEP")]
    ),
    (
        "BIL/NEFT/ICICN12025073104765344/To amma/SANTHI S/INDI",
        [("BIL", "PREFIX"), ("/", "SEP"),
         ("NEFT", "PREFIX"), ("/", "SEP"),
         ("ICICN12025073104765344", "REF_NUM"), ("/", "SEP"),
         ("To", "REMARK"), ("amma", "REMARK"), ("/", "SEP"),
         ("SANTHI", "PAYEE"), ("S", "PAYEE"), ("/", "SEP"),
         ("INDI", "BANK")]
    ),
    (
        "BIL/Personal Loan XX25011 EMI Vign",
        [("BIL", "PREFIX"), ("/", "SEP"),
         ("Personal", "PAYEE"), ("Loan", "PAYEE"), ("XX25011", "REF_NUM"),
         ("EMI", "REMARK"), ("Vign", "REMARK")]
    ),
    # ── MMT/IMPS format ────────────────────────────────────────────────────
    (
        "MMT/IMPS/528213654253/Fund Transfer/Vignesh Sa/Ind",
        [("MMT", "PREFIX"), ("/", "SEP"),
         ("IMPS", "PREFIX"), ("/", "SEP"),
         ("528213654253", "REF_NUM"), ("/", "SEP"),
         ("Fund", "REMARK"), ("Transfer", "REMARK"), ("/", "SEP"),
         ("Vignesh", "PAYEE"), ("Sa", "PAYEE"), ("/", "SEP"),
         ("Ind", "BANK")]
    ),
    # ── INF/INFT format ────────────────────────────────────────────────────
    (
        "INF/INFT/042127708371/Salary Oct25/SAAFE TECHNOLOG",
        [("INF", "PREFIX"), ("/", "SEP"),
         ("INFT", "PREFIX"), ("/", "SEP"),
         ("042127708371", "REF_NUM"), ("/", "SEP"),
         ("Salary", "REMARK"), ("Oct25", "REMARK"), ("/", "SEP"),
         ("SAAFE", "PAYEE"), ("TECHNOLOG", "PAYEE")]
    ),
    # ── ATM/POS format ─────────────────────────────────────────────────────
    (
        "RR NO:010805786465:87013369-IND LINKEDIN PGSI MUMBAI IND:",
        [("RR", "PREFIX"), ("NO", "PREFIX"), (":", "SEP"),
         ("010805786465", "REF_NUM"), (":", "SEP"),
         ("87013369", "REF_NUM"), ("-", "SEP"),
         ("IND", "OTHER"),
         ("LINKEDIN", "PAYEE"), ("PGSI", "PAYEE"),
         ("MUMBAI", "SUFFIX"), ("IND", "SUFFIX"), (":", "SEP")]
    ),
    (
        "WDL:RR NO:603023019010:00869254-PONMENI TOM MADURAI TNIN:",
        [("WDL", "PREFIX"), (":", "SEP"),
         ("RR", "PREFIX"), ("NO", "PREFIX"), (":", "SEP"),
         ("603023019010", "REF_NUM"), (":", "SEP"),
         ("00869254", "REF_NUM"), ("-", "SEP"),
         ("PONMENI", "PAYEE"), ("TOM", "PAYEE"),
         ("MADURAI", "SUFFIX"), ("TNIN", "SUFFIX"), (":", "SEP")]
    ),
    # ── NEFT TRF format (CUB) ──────────────────────────────────────────────
    (
        "HEVO TECHNOLOGIE HDFCH00761720452:",
        [("HEVO", "PAYEE"), ("TECHNOLOGIE", "PAYEE"),
         ("HDFCH00761720452", "REF_NUM"), (":", "SEP")]
    ),
    (
        "GROWW INVE",
        [("GROWW", "PAYEE"), ("INVE", "PAYEE")]
    ),
    # ── Axis Bank UPIAB/UPIAR format ──────────────────────────────────────
    (
        "UPIAB/603250424933/CR/ARUL MAR/SBIN",
        [("UPIAB", "PREFIX"), ("/", "SEP"),
         ("603250424933", "REF_NUM"), ("/", "SEP"),
         ("CR", "PREFIX"), ("/", "SEP"),
         ("ARUL", "PAYEE"), ("MAR", "PAYEE"), ("/", "SEP"),
         ("SBIN", "BANK")]
    ),
    (
        "UPIAR/603604944501/DR/ciceelia/SBIN",
        [("UPIAR", "PREFIX"), ("/", "SEP"),
         ("603604944501", "REF_NUM"), ("/", "SEP"),
         ("DR", "PREFIX"), ("/", "SEP"),
         ("ciceelia", "PAYEE"), ("/", "SEP"),
         ("SBIN", "BANK")]
    ),
    (
        "UPIAR/603701391410/DR/THE LEMO/BARB",
        [("UPIAR", "PREFIX"), ("/", "SEP"),
         ("603701391410", "REF_NUM"), ("/", "SEP"),
         ("DR", "PREFIX"), ("/", "SEP"),
         ("THE", "PAYEE"), ("LEMO", "PAYEE"), ("/", "SEP"),
         ("BARB", "BANK")]
    ),
    (
        "UPIAB/604848515910/CR/BRIANT J/SBIN",
        [("UPIAB", "PREFIX"), ("/", "SEP"),
         ("604848515910", "REF_NUM"), ("/", "SEP"),
         ("CR", "PREFIX"), ("/", "SEP"),
         ("BRIANT", "PAYEE"), ("J", "PAYEE"), ("/", "SEP"),
         ("SBIN", "BANK")]
    ),
    # ── VPS format ─────────────────────────────────────────────────────────
    (
        "VPS/KAMARAJ AKA/202511160037/531919778463/COIMBATO",
        [("VPS", "PREFIX"), ("/", "SEP"),
         ("KAMARAJ", "PAYEE"), ("AKA", "PAYEE"), ("/", "SEP"),
         ("202511160037", "REF_NUM"), ("/", "SEP"),
         ("531919778463", "REF_NUM"), ("/", "SEP"),
         ("COIMBATO", "SUFFIX")]
    ),
    # ── IPS format ─────────────────────────────────────────────────────────
    (
        "IPS/IOCL SHAN/202509191236/000000020979/COIMBATO",
        [("IPS", "PREFIX"), ("/", "SEP"),
         ("IOCL", "PAYEE"), ("SHAN", "PAYEE"), ("/", "SEP"),
         ("202509191236", "REF_NUM"), ("/", "SEP"),
         ("000000020979", "REF_NUM"), ("/", "SEP"),
         ("COIMBATO", "SUFFIX")]
    ),
]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(description: str) -> List[str]:
    """
    Split a transaction description into tokens.
    Splits on /, -, :, :: and whitespace while preserving separators.
    """
    # Normalize whitespace
    description = " ".join(description.split())
    # Split on separators, keeping them as tokens
    tokens = re.split(r'(\/|::|\-(?!\d)|:(?!:)|\s+)', description)
    # Filter empty strings, strip whitespace from tokens
    result = []
    for t in tokens:
        t = t.strip()
        if t:
            result.append(t)
    return result


# ---------------------------------------------------------------------------
# Feature extraction for CRF
# ---------------------------------------------------------------------------

def token_features(tokens: List[str], i: int) -> Dict:
    """
    Extract features for token at position i.
    CRF uses these features to learn labeling patterns.
    """
    token = tokens[i]
    token_upper = token.upper()
    token_lower = token.lower()

    features = {
        # Token identity
        'token': token_upper,
        'token_lower': token_lower,

        # Shape features
        'is_digit': token.isdigit(),
        'is_alpha': token.isalpha(),
        'is_alnum': token.isalnum(),
        'is_upper': token.isupper(),
        'is_lower': token.islower(),
        'has_at': '@' in token,
        'has_dot': '.' in token,
        'has_colon': ':' in token,
        'has_slash': '/' in token,
        'has_dash': '-' in token,
        'length': len(token),
        'length_bin': min(len(token) // 3, 5),  # 0-5 bucket

        # Digit ratio
        'digit_ratio': sum(c.isdigit() for c in token) / max(len(token), 1),

        # Known patterns
        'is_separator': token in ('/', '-', ':', '::', '|'),
        'is_bank_code': token_upper in (
            'YESB', 'HDFC', 'SBIN', 'UTIB', 'IOBA', 'UBIN', 'KKBK',
            'CNRB', 'BARB', 'FDRL', 'IDIB', 'MAHB', 'COSB', 'BKID',
            'SRCB', 'UNBA', 'MAHG', 'NSPB', 'CBIN', 'IBKL', 'INDB',
            'AIRP', 'ICIC', 'HDFC', 'AXIS', 'TMBL', 'CIUB', 'IDFB',
        ),
        'is_upi_prefix': token_upper in ('UPI', 'DR', 'CR'),
        'is_txn_prefix': token_upper in (
            'NEFT', 'IMPS', 'BIL', 'MMT', 'INF', 'INFT', 'VPS', 'IPS',
            'WDL', 'DEP', 'TFR', 'ONL', 'ATM', 'POS', 'RR', 'NO',
        ),
        'is_remark_word': token_upper in (
            'NO', 'REMARKS', 'PAYMENT', 'TRANSFER', 'FUND', 'SALARY',
            'EMI', 'LOAN', 'CREDIT', 'DEBIT', 'MANDATE', 'MANDATEREQ',
            'SENT', 'USING', 'PAID', 'VIA', 'FOR', 'TO', 'FROM',
        ),
        'is_long_alnum': len(token) > 10 and token.isalnum(),
        'starts_digit': token[0].isdigit() if token else False,
        'ends_digit': token[-1].isdigit() if token else False,

        # Position features
        'position': i,
        'position_bin': min(i // 2, 6),
        'is_first': i == 0,
        'is_second': i == 1,
        'is_last': i == len(tokens) - 1,
        'is_second_last': i == len(tokens) - 2,
    }

    # Previous token features
    if i > 0:
        prev = tokens[i - 1].upper()
        features['prev_token'] = prev
        features['prev_is_sep'] = prev in ('/', '-', ':', '::')
        features['prev_is_upi'] = prev in ('UPI', 'DR', 'CR')
        features['prev_is_ref'] = (
            tokens[i - 1].isdigit() or
            (len(tokens[i - 1]) > 10 and sum(c.isdigit() for c in tokens[i - 1]) > 5)
        )
    else:
        features['prev_token'] = '<START>'
        features['prev_is_sep'] = False
        features['prev_is_upi'] = False
        features['prev_is_ref'] = False

    # Next token features
    if i < len(tokens) - 1:
        nxt = tokens[i + 1].upper()
        features['next_token'] = nxt
        features['next_is_sep'] = nxt in ('/', '-', ':', '::')
        features['next_is_bank'] = nxt in (
            'YESB', 'HDFC', 'SBIN', 'UTIB', 'IOBA', 'UBIN', 'KKBK',
            'CNRB', 'BARB', 'FDRL', 'IDIB', 'MAHB', 'COSB', 'BKID',
        )
    else:
        features['next_token'] = '<END>'
        features['next_is_sep'] = False
        features['next_is_bank'] = False

    # Two tokens back
    if i > 1:
        features['prev2_token'] = tokens[i - 2].upper()
    else:
        features['prev2_token'] = '<START>'

    # Two tokens ahead
    if i < len(tokens) - 2:
        features['next2_token'] = tokens[i + 2].upper()
    else:
        features['next2_token'] = '<END>'

    return features


def sequence_features(tokens: List[str]) -> List[Dict]:
    """Extract features for all tokens in a sequence."""
    return [token_features(tokens, i) for i in range(len(tokens))]


# ---------------------------------------------------------------------------
# CRF Model
# ---------------------------------------------------------------------------

class CRFDescriptionParser:
    """
    CRF-based parser that extracts merchant names from transaction descriptions.
    Trained on labeled examples from CUB, SBI, ICICI, HDFC formats.
    """

    def __init__(self, model_path: str = None):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.crf = None
        self._load_or_train()

    def extract_payee(self, description: str) -> str:
        """
        Extract the merchant/payee name from a raw transaction description.

        Returns:
            Clean merchant name string, or the original description if extraction fails.
        """
        if not description:
            return "UNKNOWN"

        description = " ".join(description.split())
        tokens = tokenize(description)

        if not tokens:
            return description.upper()

        try:
            features = sequence_features(tokens)
            labels = self.crf.predict([features])[0]

            # Extract PAYEE tokens
            payee_tokens = [
                tokens[i] for i, label in enumerate(labels)
                if label == "PAYEE"
            ]

            if payee_tokens:
                return " ".join(payee_tokens).upper().strip()

            # Fallback: extract VPA tokens (often has full merchant name)
            vpa_tokens = [
                tokens[i] for i, label in enumerate(labels)
                if label == "VPA"
            ]
            if vpa_tokens:
                vpa = vpa_tokens[0]
                # Extract name from VPA (before @)
                if "@" in vpa:
                    name = vpa.split("@")[0]
                    # Remove trailing digits
                    name = re.sub(r'\d+$', '', name).strip()
                    if len(name) > 2:
                        return name.upper()

            # Last fallback: return REMARK tokens
            remark_tokens = [
                tokens[i] for i, label in enumerate(labels)
                if label == "REMARK"
            ]
            if remark_tokens:
                return " ".join(remark_tokens).upper().strip()

        except Exception as e:
            logger.warning(f"CRF extraction failed for '{description}': {e}")

        return description.upper()

    def _load_or_train(self):
        """Load trained CRF model or train a new one."""
        if self.model_path.exists():
            logger.info(f"Loading CRF parser from {self.model_path}")
            self.crf = joblib.load(self.model_path)
        else:
            logger.info("Training CRF parser...")
            self.train_and_save()

    def train_and_save(self):
        """Train the CRF model on the built-in training data."""
        try:
            import sklearn_crfsuite
        except ImportError:
            raise ImportError(
                "sklearn-crfsuite not installed. Run: pip install sklearn-crfsuite"
            )

        X_train = []
        y_train = []

        for _, labeled_tokens in TRAINING_DATA:
            tokens = [t for t, _ in labeled_tokens]
            labels = [l for _, l in labeled_tokens]
            features = sequence_features(tokens)
            X_train.append(features)
            y_train.append(labels)

        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,           # L1 regularization
            c2=0.1,           # L2 regularization
            max_iterations=200,
            all_possible_transitions=True,
        )
        self.crf.fit(X_train, y_train)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.crf, self.model_path)
        logger.info(f"CRF parser saved to {self.model_path}")

        # Quick accuracy check on training data
        correct = 0
        total = 0
        for _, labeled_tokens in TRAINING_DATA:
            tokens = [t for t, _ in labeled_tokens]
            true_labels = [l for _, l in labeled_tokens]
            features = sequence_features(tokens)
            pred_labels = self.crf.predict([features])[0]
            for t, p in zip(true_labels, pred_labels):
                if t == p:
                    correct += 1
                total += 1
        logger.info(f"CRF training accuracy: {correct}/{total} = {correct/total*100:.1f}%")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_parser_instance = None

def get_parser() -> CRFDescriptionParser:
    """Get or create the singleton CRF parser."""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = CRFDescriptionParser()
    return _parser_instance


def extract_payee(description: str) -> str:
    """Convenience function."""
    return get_parser().extract_payee(description)
