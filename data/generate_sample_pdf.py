"""
Sample Bank Statement PDF Generator
Creates realistic HDFC and SBI bank statement PDFs for testing.
Requires: reportlab  (pip install reportlab)
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Table,
        TableStyle,
        Paragraph,
        Spacer,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("reportlab not installed. Run: pip install reportlab")


# ---------------------------------------------------------------------------
# Sample transaction data
# ---------------------------------------------------------------------------

SAMPLE_TRANSACTIONS = [
    ("SWIGGY ORDER PAYMENT", -450.00, "Food"),
    ("AMAZON ONLINE PURCHASE", -2399.00, "Shopping"),
    ("IRCTC TRAIN TICKET", -1250.00, "Travel"),
    ("ELECTRICITY BILL PAYMENT", -1800.00, "Bills"),
    ("BOOKMYSHOW MOVIE TICKET", -600.00, "Entertainment"),
    ("ZOMATO FOOD DELIVERY", -380.00, "Food"),
    ("UBER RIDE PAYMENT", -220.00, "Travel"),
    ("NETFLIX SUBSCRIPTION", -649.00, "Bills"),
    ("FLIPKART ORDER PAYMENT", -1599.00, "Shopping"),
    ("SALARY CREDIT", 85000.00, "Others"),
    ("DOMINOS PIZZA ONLINE", -520.00, "Food"),
    ("AIRTEL POSTPAID BILL", -999.00, "Bills"),
    ("MAKEMYTRIP FLIGHT BOOKING", -5500.00, "Travel"),
    ("STEAM GAME PURCHASE", -1299.00, "Entertainment"),
    ("MYNTRA FASHION PURCHASE", -2100.00, "Shopping"),
    ("STARBUCKS COFFEE", -350.00, "Food"),
    ("OLA CABS RIDE", -180.00, "Travel"),
    ("AMAZON PRIME SUBSCRIPTION", -1499.00, "Bills"),
    ("PVR CINEMAS TICKET", -450.00, "Entertainment"),
    ("BIG BASKET ONLINE GROCERY", -1200.00, "Food"),
    ("NYKAA BEAUTY PRODUCTS", -899.00, "Shopping"),
    ("PETROL PUMP FUEL PURCHASE", -2500.00, "Travel"),
    ("SPOTIFY PREMIUM SUBSCRIPTION", -119.00, "Bills"),
    ("CONCERT TICKET BOOKING", -2000.00, "Entertainment"),
    ("NEFT TRANSFER RECEIVED", 10000.00, "Others"),
    ("ATM CASH WITHDRAWAL", -5000.00, "Others"),
    ("HOSPITAL BILL PAYMENT", -3500.00, "Bills"),
    ("ZEPTO GROCERY ORDER", -650.00, "Food"),
    ("RAPIDO BIKE TAXI", -80.00, "Travel"),
    ("GAMING ZONE PAYMENT", -500.00, "Entertainment"),
]


def generate_hdfc_statement(output_path: str, months: int = 2):
    """Generate a realistic HDFC bank statement PDF."""
    if not REPORTLAB_AVAILABLE:
        print("Cannot generate PDF: reportlab not installed.")
        return

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    elements = []

    # Header
    header_style = ParagraphStyle(
        "Header",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#003087"),
        spaceAfter=6,
    )
    sub_style = ParagraphStyle(
        "Sub",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
    )

    elements.append(Paragraph("HDFC BANK", header_style))
    elements.append(Paragraph("Account Statement", styles["Heading2"]))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph("Account Holder: RAHUL SHARMA", sub_style))
    elements.append(Paragraph("Account Number: XXXX XXXX 4521", sub_style))
    elements.append(Paragraph("Account Type: Savings Account", sub_style))
    elements.append(Paragraph("IFSC Code: HDFC0001234", sub_style))
    elements.append(Spacer(1, 0.5 * cm))

    # Generate transactions
    start_date = datetime.now() - timedelta(days=months * 30)
    balance = 50000.00
    transactions = []

    random.seed(42)
    for day_offset in range(months * 30):
        current_date = start_date + timedelta(days=day_offset)
        # 0–3 transactions per day
        n_txns = random.choices([0, 1, 2, 3], weights=[40, 35, 20, 5])[0]
        for _ in range(n_txns):
            txn = random.choice(SAMPLE_TRANSACTIONS)
            desc, amount, _ = txn
            # Add some variation to amounts
            amount = round(amount * random.uniform(0.8, 1.2), 2)
            balance = round(balance + amount, 2)
            transactions.append(
                (current_date.strftime("%d/%m/%Y"), desc, amount, balance)
            )

    # Table
    table_data = [["Date", "Description", "Debit (₹)", "Credit (₹)", "Balance (₹)"]]
    for date, desc, amount, bal in transactions:
        debit = f"{abs(amount):,.2f}" if amount < 0 else ""
        credit = f"{amount:,.2f}" if amount > 0 else ""
        table_data.append([date, desc[:45], debit, credit, f"{bal:,.2f}"])

    col_widths = [2.5 * cm, 8 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm]
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003087")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (1, -1), "LEFT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 0.5 * cm))
    elements.append(
        Paragraph(
            f"Closing Balance: ₹{balance:,.2f}",
            ParagraphStyle("Bold", parent=styles["Normal"], fontName="Helvetica-Bold"),
        )
    )

    doc.build(elements)
    print(f"HDFC statement saved: {output_path} ({len(transactions)} transactions)")


def generate_sbi_statement(output_path: str, months: int = 2):
    """Generate a realistic SBI bank statement PDF."""
    if not REPORTLAB_AVAILABLE:
        print("Cannot generate PDF: reportlab not installed.")
        return

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    elements = []

    header_style = ParagraphStyle(
        "Header",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#1a237e"),
        spaceAfter=6,
    )
    sub_style = ParagraphStyle(
        "Sub",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
    )

    elements.append(Paragraph("STATE BANK OF INDIA", header_style))
    elements.append(Paragraph("Account Statement", styles["Heading2"]))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph("Account Holder: PRIYA PATEL", sub_style))
    elements.append(Paragraph("Account Number: XXXX XXXX 7890", sub_style))
    elements.append(Paragraph("Account Type: Savings Bank Account", sub_style))
    elements.append(Paragraph("Branch: Mumbai Main Branch", sub_style))
    elements.append(Spacer(1, 0.5 * cm))

    # SBI uses a slightly different column layout
    start_date = datetime.now() - timedelta(days=months * 30)
    balance = 75000.00
    transactions = []

    random.seed(99)
    for day_offset in range(months * 30):
        current_date = start_date + timedelta(days=day_offset)
        n_txns = random.choices([0, 1, 2, 3], weights=[40, 35, 20, 5])[0]
        for _ in range(n_txns):
            txn = random.choice(SAMPLE_TRANSACTIONS)
            desc, amount, _ = txn
            amount = round(amount * random.uniform(0.85, 1.15), 2)
            balance = round(balance + amount, 2)
            txn_type = "DR" if amount < 0 else "CR"
            transactions.append(
                (
                    current_date.strftime("%d-%b-%Y"),
                    desc,
                    f"{abs(amount):,.2f} {txn_type}",
                    f"{balance:,.2f}",
                )
            )

    table_data = [["Txn Date", "Description", "Amount", "Balance"]]
    for date, desc, amount, bal in transactions:
        table_data.append([date, desc[:50], amount, bal])

    col_widths = [2.8 * cm, 9 * cm, 3.5 * cm, 3.5 * cm]
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a237e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (1, -1), "LEFT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e8eaf6")]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 0.5 * cm))
    elements.append(
        Paragraph(
            f"Closing Balance: ₹{balance:,.2f}",
            ParagraphStyle("Bold", parent=styles["Normal"], fontName="Helvetica-Bold"),
        )
    )

    doc.build(elements)
    print(f"SBI statement saved: {output_path} ({len(transactions)} transactions)")


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_hdfc_statement(str(output_dir / "hdfc_sample_statement.pdf"), months=2)
    generate_sbi_statement(str(output_dir / "sbi_sample_statement.pdf"), months=2)
    print("Sample PDFs generated in data/")
