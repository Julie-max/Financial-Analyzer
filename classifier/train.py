"""
Transaction Spend Classifier — Training Script
Character n-gram TF-IDF + LinearSVC
No rule-based logic, no LLM, no neural network — pure supervised ML.
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
    # ── Food ──────────────────────────────────────────────────────────────
    ("SWIGGY ORDER PAYMENT", "Food"),
    ("ZOMATO FOOD DELIVERY", "Food"),
    ("DOMINOS PIZZA ONLINE", "Food"),
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
    ("BIG BASKET ONLINE GROCERY", "Food"),
    ("GROFERS GROCERY DELIVERY", "Food"),
    ("BLINKIT QUICK COMMERCE", "Food"),
    ("ZEPTO GROCERY ORDER", "Food"),
    ("DUNZO DELIVERY FOOD", "Food"),
    ("HALDIRAMS SNACKS", "Food"),
    ("AMUL DAIRY PRODUCTS", "Food"),
    ("FRESH FRUITS VEGETABLES", "Food"),
    ("BAKERY PURCHASE", "Food"),
    ("JUICE BAR PAYMENT", "Food"),
    ("CANTEEN FOOD PURCHASE", "Food"),
    ("MESS FOOD CHARGES", "Food"),
    ("TIFFIN SERVICE PAYMENT", "Food"),
    ("MILK DELIVERY SUBSCRIPTION", "Food"),
    ("BURGER KING PAYMENT", "Food"),
    ("WENDY'S RESTAURANT", "Food"),
    ("CHIPOTLE FOOD ORDER", "Food"),
    ("INSTACART GROCERY", "Food"),
    ("WHOLE FOODS MARKET", "Food"),
    ("TRADER JOES PURCHASE", "Food"),
    ("WALMART GROCERY", "Food"),
    ("COSTCO FOOD PURCHASE", "Food"),
    ("NANDOS RESTAURANT", "Food"),
    ("PRET A MANGER CAFE", "Food"),
    ("LEON RESTAURANT", "Food"),
    ("JUST EAT ORDER", "Food"),
    ("DELIVEROO FOOD DELIVERY", "Food"),
    ("UBER EATS ORDER", "Food"),
    ("DOORDASH FOOD DELIVERY", "Food"),
    ("GRUBHUB ORDER PAYMENT", "Food"),
    ("POSTMATES DELIVERY", "Food"),
    ("FOOD PANDA ORDER", "Food"),
    ("LUNCHBOX MEAL DELIVERY", "Food"),
    ("FRESHMENU FOOD ORDER", "Food"),
    ("FAASOS FOOD DELIVERY", "Food"),
    ("BOX8 MEAL ORDER", "Food"),
    ("BEHROUZ BIRYANI ORDER", "Food"),

    # ── Travel ────────────────────────────────────────────────────────────
    ("IRCTC TRAIN TICKET BOOKING", "Travel"),
    ("MAKEMYTRIP FLIGHT BOOKING", "Travel"),
    ("GOIBIBO HOTEL BOOKING", "Travel"),
    ("OYO ROOMS BOOKING", "Travel"),
    ("UBER RIDE PAYMENT", "Travel"),
    ("OLA CABS RIDE", "Travel"),
    ("RAPIDO BIKE TAXI", "Travel"),
    ("INDIGO AIRLINES TICKET", "Travel"),
    ("AIR INDIA FLIGHT BOOKING", "Travel"),
    ("SPICEJET TICKET PURCHASE", "Travel"),
    ("VISTARA AIRLINES BOOKING", "Travel"),
    ("REDBUS TICKET BOOKING", "Travel"),
    ("ABHIBUS TICKET PURCHASE", "Travel"),
    ("METRO CARD RECHARGE", "Travel"),
    ("FASTAG TOLL RECHARGE", "Travel"),
    ("PETROL PUMP FUEL PURCHASE", "Travel"),
    ("HP PETROL STATION", "Travel"),
    ("INDIAN OIL FUEL PAYMENT", "Travel"),
    ("BHARAT PETROLEUM FUEL", "Travel"),
    ("CAR RENTAL PAYMENT", "Travel"),
    ("TAXI SERVICE PAYMENT", "Travel"),
    ("AUTO RICKSHAW PAYMENT", "Travel"),
    ("LYFT RIDE PAYMENT", "Travel"),
    ("GRAB TAXI BOOKING", "Travel"),
    ("AIRBNB ACCOMMODATION", "Travel"),
    ("BOOKING.COM HOTEL", "Travel"),
    ("EXPEDIA TRAVEL BOOKING", "Travel"),
    ("TRIVAGO HOTEL BOOKING", "Travel"),
    ("AGODA HOTEL PAYMENT", "Travel"),
    ("EMIRATES AIRLINES TICKET", "Travel"),
    ("BRITISH AIRWAYS BOOKING", "Travel"),
    ("DELTA AIRLINES TICKET", "Travel"),
    ("UNITED AIRLINES BOOKING", "Travel"),
    ("SOUTHWEST AIRLINES TICKET", "Travel"),
    ("AMTRAK TRAIN TICKET", "Travel"),
    ("GREYHOUND BUS TICKET", "Travel"),
    ("NATIONAL RAIL TICKET", "Travel"),
    ("EUROSTAR TRAIN BOOKING", "Travel"),
    ("PARKING CHARGES PAYMENT", "Travel"),
    ("TOLL PLAZA PAYMENT", "Travel"),
    ("VEHICLE MAINTENANCE SERVICE", "Travel"),
    ("CAR WASH PAYMENT", "Travel"),
    ("TYRE PUNCTURE REPAIR", "Travel"),
    ("BIKE SERVICE PAYMENT", "Travel"),
    ("SCOOTER RENTAL PAYMENT", "Travel"),
    ("CYCLE RENTAL PAYMENT", "Travel"),
    ("FERRY TICKET BOOKING", "Travel"),
    ("CRUISE BOOKING PAYMENT", "Travel"),
    ("TRAVEL INSURANCE PREMIUM", "Travel"),
    ("VISA APPLICATION FEE", "Travel"),

    # ── Shopping ──────────────────────────────────────────────────────────
    ("AMAZON ONLINE PURCHASE", "Shopping"),
    ("FLIPKART ORDER PAYMENT", "Shopping"),
    ("MYNTRA FASHION PURCHASE", "Shopping"),
    ("AJIO CLOTHING PURCHASE", "Shopping"),
    ("NYKAA BEAUTY PRODUCTS", "Shopping"),
    ("MEESHO ONLINE SHOPPING", "Shopping"),
    ("SNAPDEAL PURCHASE", "Shopping"),
    ("SHOPCLUES ORDER", "Shopping"),
    ("RELIANCE DIGITAL PURCHASE", "Shopping"),
    ("CROMA ELECTRONICS STORE", "Shopping"),
    ("VIJAY SALES ELECTRONICS", "Shopping"),
    ("APPLE STORE PURCHASE", "Shopping"),
    ("SAMSUNG STORE PAYMENT", "Shopping"),
    ("IKEA FURNITURE PURCHASE", "Shopping"),
    ("PEPPERFRY FURNITURE ORDER", "Shopping"),
    ("URBAN LADDER FURNITURE", "Shopping"),
    ("LIFESTYLE FASHION STORE", "Shopping"),
    ("WESTSIDE CLOTHING STORE", "Shopping"),
    ("ZARA FASHION PURCHASE", "Shopping"),
    ("H&M CLOTHING PURCHASE", "Shopping"),
    ("UNIQLO APPAREL PURCHASE", "Shopping"),
    ("MARKS SPENCER PURCHASE", "Shopping"),
    ("PANTALOONS CLOTHING", "Shopping"),
    ("MAX FASHION PURCHASE", "Shopping"),
    ("SHOPPERS STOP PURCHASE", "Shopping"),
    ("CENTRAL MALL PURCHASE", "Shopping"),
    ("EBAY ONLINE PURCHASE", "Shopping"),
    ("ETSY HANDMADE PURCHASE", "Shopping"),
    ("WALMART RETAIL PURCHASE", "Shopping"),
    ("TARGET STORE PURCHASE", "Shopping"),
    ("BEST BUY ELECTRONICS", "Shopping"),
    ("HOME DEPOT PURCHASE", "Shopping"),
    ("LOWES HOME IMPROVEMENT", "Shopping"),
    ("MACYS DEPARTMENT STORE", "Shopping"),
    ("NORDSTROM PURCHASE", "Shopping"),
    ("GAP CLOTHING STORE", "Shopping"),
    ("OLD NAVY PURCHASE", "Shopping"),
    ("FOREVER 21 CLOTHING", "Shopping"),
    ("PRIMARK FASHION STORE", "Shopping"),
    ("NEXT CLOTHING PURCHASE", "Shopping"),
    ("JOHN LEWIS PURCHASE", "Shopping"),
    ("BOOTS PHARMACY PURCHASE", "Shopping"),
    ("CHEMIST WAREHOUSE PURCHASE", "Shopping"),
    ("DECATHLON SPORTS GOODS", "Shopping"),
    ("NIKE STORE PURCHASE", "Shopping"),
    ("ADIDAS STORE PAYMENT", "Shopping"),
    ("PUMA FOOTWEAR PURCHASE", "Shopping"),
    ("REEBOK SPORTS PURCHASE", "Shopping"),
    ("SKECHERS FOOTWEAR", "Shopping"),
    ("BATA SHOES PURCHASE", "Shopping"),

    # ── Bills ─────────────────────────────────────────────────────────────
    ("ELECTRICITY BILL PAYMENT", "Bills"),
    ("WATER BILL PAYMENT", "Bills"),
    ("GAS BILL PAYMENT", "Bills"),
    ("BROADBAND INTERNET BILL", "Bills"),
    ("MOBILE RECHARGE PAYMENT", "Bills"),
    ("DTH RECHARGE TATASKY", "Bills"),
    ("AIRTEL POSTPAID BILL", "Bills"),
    ("JIO MOBILE BILL PAYMENT", "Bills"),
    ("VODAFONE BILL PAYMENT", "Bills"),
    ("BSNL LANDLINE BILL", "Bills"),
    ("HOUSE RENT PAYMENT", "Bills"),
    ("SOCIETY MAINTENANCE CHARGES", "Bills"),
    ("INSURANCE PREMIUM PAYMENT", "Bills"),
    ("LIC PREMIUM PAYMENT", "Bills"),
    ("HEALTH INSURANCE PREMIUM", "Bills"),
    ("CAR INSURANCE RENEWAL", "Bills"),
    ("HOME LOAN EMI PAYMENT", "Bills"),
    ("PERSONAL LOAN EMI", "Bills"),
    ("CREDIT CARD BILL PAYMENT", "Bills"),
    ("MUNICIPAL TAX PAYMENT", "Bills"),
    ("PROPERTY TAX PAYMENT", "Bills"),
    ("INCOME TAX PAYMENT", "Bills"),
    ("GST PAYMENT", "Bills"),
    ("SCHOOL FEES PAYMENT", "Bills"),
    ("COLLEGE TUITION FEES", "Bills"),
    ("HOSPITAL BILL PAYMENT", "Bills"),
    ("MEDICAL BILL PAYMENT", "Bills"),
    ("PHARMACY MEDICINE PURCHASE", "Bills"),
    ("DOCTOR CONSULTATION FEE", "Bills"),
    ("DIAGNOSTIC LAB PAYMENT", "Bills"),
    ("NETFLIX SUBSCRIPTION", "Bills"),
    ("AMAZON PRIME SUBSCRIPTION", "Bills"),
    ("SPOTIFY PREMIUM SUBSCRIPTION", "Bills"),
    ("YOUTUBE PREMIUM PAYMENT", "Bills"),
    ("HOTSTAR SUBSCRIPTION", "Bills"),
    ("APPLE SUBSCRIPTION PAYMENT", "Bills"),
    ("GOOGLE ONE STORAGE PLAN", "Bills"),
    ("MICROSOFT 365 SUBSCRIPTION", "Bills"),
    ("ADOBE CREATIVE CLOUD", "Bills"),
    ("GITHUB SUBSCRIPTION", "Bills"),
    ("DOMAIN HOSTING RENEWAL", "Bills"),
    ("VPS SERVER PAYMENT", "Bills"),
    ("AWS CLOUD BILL", "Bills"),
    ("AZURE SUBSCRIPTION BILL", "Bills"),
    ("GOOGLE CLOUD PAYMENT", "Bills"),
    ("CABLE TV BILL PAYMENT", "Bills"),
    ("NEWSPAPER SUBSCRIPTION", "Bills"),
    ("MAGAZINE SUBSCRIPTION", "Bills"),
    ("CLUB MEMBERSHIP FEE", "Bills"),
    ("GYM MEMBERSHIP PAYMENT", "Bills"),

    # ── Entertainment ─────────────────────────────────────────────────────
    ("BOOKMYSHOW MOVIE TICKET", "Entertainment"),
    ("PVR CINEMAS TICKET", "Entertainment"),
    ("INOX MOVIES TICKET", "Entertainment"),
    ("CINEPOLIS TICKET BOOKING", "Entertainment"),
    ("CARNIVAL CINEMAS TICKET", "Entertainment"),
    ("GAMING ZONE PAYMENT", "Entertainment"),
    ("STEAM GAME PURCHASE", "Entertainment"),
    ("PLAYSTATION STORE PURCHASE", "Entertainment"),
    ("XBOX GAME PASS PAYMENT", "Entertainment"),
    ("NINTENDO ESHOP PURCHASE", "Entertainment"),
    ("EPIC GAMES STORE PURCHASE", "Entertainment"),
    ("GOOGLE PLAY GAMES PURCHASE", "Entertainment"),
    ("APPLE ARCADE SUBSCRIPTION", "Entertainment"),
    ("CONCERT TICKET BOOKING", "Entertainment"),
    ("LIVE EVENT TICKET PURCHASE", "Entertainment"),
    ("SPORTS EVENT TICKET", "Entertainment"),
    ("IPL MATCH TICKET", "Entertainment"),
    ("THEME PARK ENTRY TICKET", "Entertainment"),
    ("ZOO ENTRY TICKET", "Entertainment"),
    ("MUSEUM ENTRY TICKET", "Entertainment"),
    ("AMUSEMENT PARK TICKET", "Entertainment"),
    ("BOWLING ALLEY PAYMENT", "Entertainment"),
    ("LASER TAG GAME PAYMENT", "Entertainment"),
    ("ESCAPE ROOM BOOKING", "Entertainment"),
    ("KARAOKE BAR PAYMENT", "Entertainment"),
    ("NIGHTCLUB ENTRY PAYMENT", "Entertainment"),
    ("BAR DRINKS PAYMENT", "Entertainment"),
    ("PUB PAYMENT", "Entertainment"),
    ("COMEDY SHOW TICKET", "Entertainment"),
    ("THEATRE PLAY TICKET", "Entertainment"),
    ("OPERA TICKET BOOKING", "Entertainment"),
    ("BOOK PURCHASE AMAZON", "Entertainment"),
    ("KINDLE EBOOK PURCHASE", "Entertainment"),
    ("AUDIBLE AUDIOBOOK", "Entertainment"),
    ("SPOTIFY MUSIC STREAMING", "Entertainment"),
    ("APPLE MUSIC SUBSCRIPTION", "Entertainment"),
    ("GAANA MUSIC SUBSCRIPTION", "Entertainment"),
    ("JIOSAAVN MUSIC PLAN", "Entertainment"),
    ("WYNK MUSIC SUBSCRIPTION", "Entertainment"),
    ("TWITCH SUBSCRIPTION PAYMENT", "Entertainment"),
    ("YOUTUBE CHANNEL MEMBERSHIP", "Entertainment"),
    ("PATREON CREATOR SUPPORT", "Entertainment"),
    ("FANTASY SPORTS PAYMENT", "Entertainment"),
    ("DREAM11 CONTEST ENTRY", "Entertainment"),
    ("MPL GAME ENTRY FEE", "Entertainment"),
    ("LUDO KING PURCHASE", "Entertainment"),
    ("CHESS.COM SUBSCRIPTION", "Entertainment"),
    ("DUOLINGO PLUS SUBSCRIPTION", "Entertainment"),
    ("COURSERA COURSE PURCHASE", "Entertainment"),
    ("UDEMY COURSE PURCHASE", "Entertainment"),

    # ── Others ────────────────────────────────────────────────────────────
    ("SALARY CREDIT FROM EMPLOYER", "Others"),
    ("NEFT TRANSFER RECEIVED", "Others"),
    ("IMPS TRANSFER SENT", "Others"),
    ("UPI PAYMENT TRANSFER", "Others"),
    ("RTGS FUND TRANSFER", "Others"),
    ("BANK CHARGES DEDUCTED", "Others"),
    ("ATM CASH WITHDRAWAL", "Others"),
    ("CASH DEPOSIT AT BRANCH", "Others"),
    ("CHEQUE DEPOSIT CLEARING", "Others"),
    ("DIVIDEND CREDIT RECEIVED", "Others"),
    ("INTEREST CREDIT SAVINGS", "Others"),
    ("FIXED DEPOSIT MATURITY", "Others"),
    ("MUTUAL FUND PURCHASE SIP", "Others"),
    ("STOCK MARKET PURCHASE", "Others"),
    ("ZERODHA BROKERAGE CHARGE", "Others"),
    ("GROWW INVESTMENT PAYMENT", "Others"),
    ("UPSTOX TRADING CHARGE", "Others"),
    ("GOLD PURCHASE PAYMENT", "Others"),
    ("JEWELLERY STORE PURCHASE", "Others"),
    ("DONATION CHARITY PAYMENT", "Others"),
    ("TEMPLE DONATION PAYMENT", "Others"),
    ("GIFT CARD PURCHASE", "Others"),
    ("AMAZON PAY WALLET LOAD", "Others"),
    ("PAYTM WALLET RECHARGE", "Others"),
    ("PHONEPE WALLET TOPUP", "Others"),
    ("GOOGLE PAY TRANSFER", "Others"),
    ("REFUND CREDIT RECEIVED", "Others"),
    ("CASHBACK CREDIT", "Others"),
    ("REWARD POINTS REDEMPTION", "Others"),
    ("LOAN DISBURSEMENT CREDIT", "Others"),
    ("ADVANCE SALARY CREDIT", "Others"),
    ("FREELANCE PAYMENT RECEIVED", "Others"),
    ("CONSULTING FEE RECEIVED", "Others"),
    ("RENTAL INCOME CREDIT", "Others"),
    ("GOVERNMENT SUBSIDY CREDIT", "Others"),
    ("TAX REFUND CREDIT", "Others"),
    ("INSURANCE CLAIM SETTLEMENT", "Others"),
    ("LEGAL FEE PAYMENT", "Others"),
    ("NOTARY CHARGES PAYMENT", "Others"),
    ("COURIER SERVICE PAYMENT", "Others"),
    ("POSTAL CHARGES PAYMENT", "Others"),
    ("PRINTING CHARGES PAYMENT", "Others"),
    ("STATIONERY PURCHASE", "Others"),
    ("HARDWARE TOOLS PURCHASE", "Others"),
    ("PLUMBER SERVICE PAYMENT", "Others"),
    ("ELECTRICIAN SERVICE CHARGE", "Others"),
    ("CARPENTER SERVICE PAYMENT", "Others"),
    ("MAID SALARY PAYMENT", "Others"),
    ("DRIVER SALARY PAYMENT", "Others"),
    ("SECURITY DEPOSIT PAYMENT", "Others"),

    # ── UPI-format real bank statement patterns ────────────────────────────
    # Food
    ("MK FOODS FOOD", "Food"),
    ("SWIGGY NO REMARKS", "Food"),
    ("ZOMATO LIM ZOMATO PAY", "Food"),
    ("EACHANARI FOOD", "Food"),
    ("EACHANARI SNACKS", "Food"),
    ("KARUPPAIYA DINNER", "Food"),
    ("DEEPA DHAR LUNCH", "Food"),
    ("VARA MILAG LUNCH", "Food"),
    ("VARA MILAG FOOD", "Food"),
    ("THATS Y FO NO REMARKS", "Food"),
    ("AMARAVATHI DINNER", "Food"),
    ("THIRD WAVE PAYMENT", "Food"),
    ("NEW WOODLA DINNER", "Food"),
    ("LAYALEE GR BREAKFAST", "Food"),
    ("MAHENDRA P LUNCH", "Food"),
    ("GURUSADUPA LUNCH", "Food"),
    ("T STANES A TEA", "Food"),
    ("AMARA ENTE RAVAI", "Food"),
    ("ANANDAKUMA FRUITS", "Food"),
    ("MARIMUTHU FLOWERS", "Food"),
    ("RAMESHKUMA THAKKALI", "Food"),
    ("BASIL FNB MUDDE", "Food"),
    ("MR VEGETAB ONION", "Food"),
    ("INIYAAS RE TEA", "Food"),
    ("DIVYANAND LUNCH", "Food"),
    ("SHREE ANAN SNACKS", "Food"),
    ("A BALAMURU DINNER", "Food"),
    ("AIRPORT SR COFFEE", "Food"),
    ("AIRPORT SR PUFF", "Food"),
    ("AIRPORT SR ROSE MILK", "Food"),
    ("A GUNASEKA SNACKS", "Food"),
    ("ANANDA HOT PAYMENT", "Food"),
    ("ANNAPOORAN SNACKS", "Food"),
    ("CANTEEN CO NO REMARKS", "Food"),
    ("TEA TIME TEA", "Food"),
    ("MALGUDI LUNCH", "Food"),

    # Travel
    ("OYO ROOMS BOOKING", "Travel"),
    ("MAKEMYTRIP PAYMENT", "Travel"),
    ("MAKE MY TR PAYMENT", "Travel"),
    ("IRCTC WEB NO REMARKS", "Travel"),
    ("IRCTC NRM NO REMARKS", "Travel"),
    ("PAYTM TRAV SENT USING", "Travel"),
    ("SENTHIL CAB", "Travel"),
    ("NAGARAJ AUTO", "Travel"),
    ("KALAI SELV AUTO", "Travel"),
    ("KARTHIKEYA RIDE", "Travel"),
    ("THASTHAGEE AUTO", "Travel"),
    ("ARUL AYYAP CAB", "Travel"),
    ("JAWAHAR CAB", "Travel"),
    ("TOWN HOUSE ROOM STAY", "Travel"),
    ("MUHAMMED A CAB", "Travel"),
    ("DHARMARAJ RIDE", "Travel"),
    ("VIGNESHWAR AIRPORT", "Travel"),
    ("IYYAPPAN G CAN", "Travel"),
    ("VPS KAMARAJ", "Travel"),
    ("CMRL AIRPO RDS", "Travel"),
    ("INDIAN RAI PAYMENT", "Travel"),
    ("BHIMANNA H CAB RIDE", "Travel"),
    ("MURUGESAN CAB", "Travel"),
    ("RANGAPPA G RIDE", "Travel"),
    ("MANIKANDAN AUTO", "Travel"),
    ("BALAN A AUTO", "Travel"),
    ("KANAGARAJ CAB", "Travel"),
    ("LALAN KUMA RIDE", "Travel"),
    ("THIYAGARAJ RIDE", "Travel"),
    ("AMARSINGH CAB", "Travel"),
    ("FURKAN FUR AUTO", "Travel"),
    ("DINESH SAN AUTO", "Travel"),
    ("VINOTH HAR CAB", "Travel"),
    ("UMAPATHY J AUTO", "Travel"),
    ("MR T MOSE CAB", "Travel"),
    ("SUMATHI AUTO", "Travel"),
    ("ASMA SHOP AUTO", "Travel"),
    ("DEIVAMANI RIDE", "Travel"),
    ("BALAMURUGA CAN", "Travel"),
    ("IOCL SHAN COIMBATO", "Travel"),
    ("GEETA SOOR EARLY CHEC", "Travel"),
    ("AIR INDIA AIRINDIA", "Travel"),

    # Bills
    ("JIOFIBER P PAYMENT", "Bills"),
    ("JIO POSTPA NO REMARKS", "Bills"),
    ("TANGEDCO TNPDCL POR", "Bills"),
    ("ICICI BANK CREDIT CA", "Bills"),
    ("SBI CARDS PAY", "Bills"),
    ("PERSONAL LOAN EMI", "Bills"),
    ("LINKEDIN MANDATEREQ", "Bills"),
    ("SPOTIFY IN MANDATEREQ", "Bills"),
    ("APPLE MEDI UPI MANDAT", "Bills"),
    ("AWS INDIA AMAZON WEB", "Bills"),
    ("LIFE INSUR SENT USING", "Bills"),
    ("CBDT UPIINTENT", "Bills"),
    ("ECHALLAN T ECHALLAN", "Bills"),
    ("GOOGLE IND UPI", "Bills"),
    ("EUREKA FOR EUREKAFORB", "Bills"),

    # Shopping
    ("GHARSOAPS PAY", "Shopping"),
    ("DUMMY NAME EARBUDS", "Shopping"),
    ("MR ROHAN D SLIPPERS", "Shopping"),
    ("SUPULAXMI BLOUSE WOR", "Shopping"),
    ("KOVAI PAZH PAYMENT", "Shopping"),
    ("APPANASAMY TONER", "Shopping"),
    ("AMARA ENTE GROCERIES", "Shopping"),
    ("RUPTUB SOL TREEBO", "Shopping"),
    ("TAMILMARAN CORE CUTTI", "Shopping"),
    ("SIVAKUMAR MERCHANT", "Shopping"),
    ("G RATHINAK STORE PIPE", "Shopping"),
    ("G RATHINAK PIPES", "Shopping"),
    ("ROY Y PLY MATERI", "Shopping"),
    ("ROY Y PLYWORLD", "Shopping"),
    ("RK ELECTRO CCTV", "Shopping"),
    ("SANTHAKUMA AH COLORS", "Shopping"),
    ("KAMALA ELE UPI", "Shopping"),
    ("KARTHIK SU GLASS BALA", "Shopping"),

    # Entertainment
    ("ZERODHA FU PAYMENT", "Entertainment"),
    ("ICCL ZEROD TFES", "Entertainment"),
    ("ZERODHA BROKING", "Entertainment"),
    ("MUSEUM TICKET", "Entertainment"),
    ("MUSEUM3 TICKET", "Entertainment"),
    ("LE GRACE CHENNAI", "Entertainment"),
    ("VPS DENNIS STOR COIMBATO", "Entertainment"),
    ("GEM HOSPITA COIMBATO", "Entertainment"),
    ("ICCL ZEROD ZERODHA", "Entertainment"),

    # Others
    ("PENTAFOX TECHNOLOGIES SALARY", "Others"),
    ("SAAFE TECHNOLO SALARY", "Others"),
    ("PENTAFOX TECHNOLOGIES PVT LTD", "Others"),
    ("FUND TRANSFER VIGNESH", "Others"),
    ("MMT IMPS FUND TRANSFER", "Others"),
    ("BIL INFT HOUSE EXPENSE", "Others"),
    ("BIL INFT FAMILY", "Others"),
    ("HOME LOAN KVLPM", "Others"),
    ("RAMYA GOWR PAYMENT", "Others"),
    ("CHANDRASEK PAYMENT", "Others"),
    ("MRS NEELAV PAYMENT", "Others"),
    ("VIJAYAN RA UPI", "Others"),
    ("SUSHIN S PAYMENT", "Others"),
    ("SRIVIDYA S PAYMENT", "Others"),
    ("VARDHINI S PAYMENT", "Others"),
    ("KARPAGAM ADVANCE", "Others"),
    ("ZERODHA FUND HOUSE PAYMENT", "Others"),
    ("RVH CAPITAL INVESTMENT", "Others"),
    ("DEBT REPAY", "Others"),
    ("PL EMI SRIRAM", "Others"),
    ("SRIRAM M EMI", "Others"),
    ("INTEREST CREDIT", "Others"),
    ("INT PD SAVINGS", "Others"),
    ("BHIMCASHBACK BHIMCASHBA", "Others"),
    ("NPCI BHIM BHIMCASHBA", "Others"),
    ("SARAVANAN PAYMENT", "Others"),
    ("SURENDAR S PAYMENT", "Others"),
    ("SURESH C PAYMENT", "Others"),
    ("RASOOL MYD PAYMENT", "Others"),
    ("VALLAVARAJ PAYMENT", "Others"),
    ("PARAMESH PAYMENT", "Others"),
    ("ALEXANDER PAYMENT", "Others"),
    ("RATNAA SHR PAYMENT", "Others"),
    ("D NAVEEN K PAYMENT", "Others"),
    ("MANIKANDAN PAYMENT", "Others"),
    ("THIRUMA PAYMENT", "Others"),
    ("SASIKUMAR PAYMENT", "Others"),
    ("VAIDHIYANA PAYMENT", "Others"),
    ("KARTHIK P PAYMENT", "Others"),
    ("SUBRAMANIA NO REMARKS", "Others"),
    ("NAGARAJA COURIER", "Others"),
    ("B JAGADEES COURIER", "Others"),
    ("PRAVEEN KU GFF EXPENS", "Others"),
    ("IMRAN F PAYMENT", "Others"),
    ("MR VIMAL R TRANSPORT", "Others"),
    ("VADIRAJASH CASH TO DR", "Others"),
    ("VAISHNAVI CRACKERS", "Others"),
    ("PENTAFOX T CRACKERS", "Others"),
    ("SHADAB AHM UPI", "Others"),
    ("MR JOHN PR CAN", "Others"),
    ("GHADAGE OM CAB", "Others"),
    ("KATHIRESAN TAXI", "Others"),
    ("MOHAMMAD M AUTO", "Others"),
    ("AKEEL AUTO", "Others"),
    ("VETRI VENT PAYMENT", "Others"),
    ("DANIEL KIN PAYMENT", "Others"),
    ("LOGESHKUMA TAXI", "Others"),
    ("LOGESHWHAR PAYMENT", "Others"),
    ("SHANKAR B BALANCE LO", "Others"),
    ("GOKUL SP VOUCHERS", "Others"),
    ("SARATHI JIO RELOCA", "Others"),
    ("PAVAN KUMA VADA", "Others"),
    ("MR GAUTAM IDLI", "Others"),
    ("KADAM HEMA CAN", "Others"),
    ("MOHANRAJ G PAYMENT", "Others"),
    ("NAFISA BEG RIDE", "Others"),
    ("DINESH DUR AUTO", "Others"),
    ("VENKATESH RIDE", "Others"),
    ("S PREMKUMA AUTO", "Others"),
    ("DURAIMURUG PAYMENT", "Others"),
    ("BARANI BAK NO REMARKS", "Others"),
    ("THILAGAVAT NO REMARKS", "Others"),
    ("RANI S PARKING", "Others"),
    ("THASTHAGEE AUTO FARE", "Others"),
    ("BHIMANNA H CAB RIDE", "Others"),
    ("PRAMOD MAH AUTO", "Others"),
    ("PAVAN AUTO", "Others"),
    ("MD JAKIRUL RIDE", "Others"),
    ("RAM MAITY RIDE", "Others"),
    ("SRI VISHNU NO REMARKS", "Others"),
    ("DIRECTORAT PAYMENT", "Others"),
    ("ARAVINDARA ELECTRICIA", "Others"),
    ("KARTHIKEYA PAYMENT", "Others"),
    ("SHANMUGANA AUTO RIDE", "Others"),

    # ── VPA signal training examples ──────────────────────────────────────
    # These teach the classifier that VPA_PERSONAL → Others (P2P transfer)
    # and VPA_MERCHANT → use description for category

    # Personal VPA → Others (person-to-person UPI payments)
    ("ARUL MAR VPA_PERSONAL", "Others"),
    ("JULIESCH VPA_PERSONAL", "Others"),
    ("B MUHAMM VPA_PERSONAL", "Others"),
    ("MR KALID VPA_PERSONAL", "Others"),
    ("CICEELIA VPA_PERSONAL", "Others"),
    ("BRIANT J VPA_PERSONAL", "Others"),
    ("JEYANTHI VPA_PERSONAL", "Others"),
    ("SUJIT BR VPA_PERSONAL", "Others"),
    ("RUPALI S VPA_PERSONAL", "Others"),
    ("VISHAL B VPA_PERSONAL", "Others"),
    ("SANJAI VPA_PERSONAL", "Others"),
    ("RAMYA GOWR VPA_PERSONAL", "Others"),
    ("CHANDRASEK VPA_PERSONAL", "Others"),
    ("MRS NEELAV VPA_PERSONAL", "Others"),
    ("VIJAYAN RA VPA_PERSONAL", "Others"),
    ("SRIVIDYA S VPA_PERSONAL", "Others"),
    ("VARDHINI S VPA_PERSONAL", "Others"),
    ("SURENDAR S VPA_PERSONAL", "Others"),
    ("MANIKANDAN VPA_PERSONAL", "Others"),
    ("THIRUMA VPA_PERSONAL", "Others"),
    ("SASIKUMAR VPA_PERSONAL", "Others"),
    ("NAGARAJA VPA_PERSONAL", "Others"),
    ("PRAVEEN KU VPA_PERSONAL", "Others"),
    ("DANIEL KIN VPA_PERSONAL", "Others"),
    ("VETRI VENT VPA_PERSONAL", "Others"),
    ("MUHAMMED A VPA_PERSONAL", "Others"),
    ("AMARSINGH VPA_PERSONAL", "Others"),
    ("DHARMARAJ VPA_PERSONAL", "Others"),
    ("KANAGARAJ VPA_PERSONAL", "Others"),
    ("LALAN KUMA VPA_PERSONAL", "Others"),
    ("THIYAGARAJ VPA_PERSONAL", "Others"),
    ("NAFISA BEG VPA_PERSONAL", "Others"),
    ("DINESH DUR VPA_PERSONAL", "Others"),
    ("VENKATESH VPA_PERSONAL", "Others"),
    ("S PREMKUMA VPA_PERSONAL", "Others"),
    ("DURAIMURUG VPA_PERSONAL", "Others"),
    ("BARANI BAK VPA_PERSONAL", "Others"),
    ("THILAGAVAT VPA_PERSONAL", "Others"),
    ("RANI S VPA_PERSONAL", "Others"),
    ("BHIMANNA H VPA_PERSONAL", "Others"),
    ("PRAMOD MAH VPA_PERSONAL", "Others"),
    ("PAVAN VPA_PERSONAL", "Others"),
    ("MD JAKIRUL VPA_PERSONAL", "Others"),
    ("RAM MAITY VPA_PERSONAL", "Others"),
    ("SRI VISHNU VPA_PERSONAL", "Others"),

    # Merchant VPA → correct category (VPA has @domain)
    ("SWIGGY VPA_MERCHANT", "Food"),
    ("ZOMATO VPA_MERCHANT", "Food"),
    ("MAKEMYTR VPA_MERCHANT", "Travel"),
    ("MAKEMYTRIP VPA_MERCHANT", "Travel"),
    ("IRCTC VPA_MERCHANT", "Travel"),
    ("BOOKMYSH VPA_MERCHANT", "Entertainment"),
    ("NETFLIX VPA_MERCHANT", "Bills"),
    ("SPOTIFY VPA_MERCHANT", "Bills"),
    ("LINKEDIN VPA_MERCHANT", "Bills"),
    ("AMAZON VPA_MERCHANT", "Shopping"),
    ("FLIPKART VPA_MERCHANT", "Shopping"),
    ("JIOFIBER VPA_MERCHANT", "Bills"),
    ("AIRTEL VPA_MERCHANT", "Bills"),
    ("ZEPTO VPA_MERCHANT", "Food"),
    ("BLINKIT VPA_MERCHANT", "Food"),
    ("REDBUS VPA_MERCHANT", "Travel"),
    ("UBER VPA_MERCHANT", "Travel"),
    ("OLA VPA_MERCHANT", "Travel"),

    # QR code VPA → use description (small vendors, ambiguous)
    ("AMMAN CO VPA_QR", "Food"),
    ("WINE HIL VPA_QR", "Food"),
    ("CHAI FI VPA_QR", "Food"),
    ("HOTEL GO VPA_QR", "Food"),
    ("THE LEMO VPA_QR", "Food"),
    ("KALYANA VPA_QR", "Food"),
    ("MANIKODI VPA_QR", "Others"),
]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Normalise a transaction description for ML feature extraction.
    - Lowercase
    - Remove special characters (keep alphanumeric + spaces)
    - Collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_pipeline(model_type: str = "svm") -> Pipeline:
    """
    Build a scikit-learn Pipeline with character n-gram TF-IDF + LinearSVC.

    Uses character-level n-grams (2-5 chars) instead of word n-grams.
    This handles truncated merchant names like BOOKMYSH, MAKEMYTR, SWIGGYSTOR
    because character sequences overlap with the full names in training data.

    Args:
        model_type: 'svm' for LinearSVC (default), 'logreg' for Logistic Regression.
    """
    # Character n-gram TF-IDF — the key improvement
    # analyzer='char_wb' includes word boundaries, better than 'char' for short text
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),       # 2 to 5 character n-grams
        min_df=1,
        max_features=50_000,
        sublinear_tf=True,        # log(1+tf) scaling
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
        # LinearSVC is faster and works better with high-dimensional sparse features
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

    Args:
        data: List of (description, category) tuples. Defaults to TRAINING_DATA.
        model_type: 'logreg' or 'svm'.
        model_dir: Directory to save model artifacts.
        cv_folds: Number of cross-validation folds.

    Returns:
        Dict with training metrics.
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
