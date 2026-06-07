"""
Financial Analyzer — Streamlit UI
Upload bank statement PDFs, extract transactions and classify spend.
"""

import sys
import os
import json
import logging
import tempfile
from pathlib import Path

# Add project root to path so imports work when running from app/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.pipeline import FinancialPipeline
from utils.insights import (
    compute_category_summary,
    compute_monthly_summary,
    compute_monthly_totals,
    generate_insights,
    get_top_merchants,
)
from classifier.train import train as train_classifier, CATEGORIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.25rem;
    }
    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        font-size: 0.95rem;
    }
    .stAlert { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💰 Financial Analyzer")
    st.markdown("*Offline bank statement analysis*")
    st.markdown("---")

    # Show current data status
    if "transactions" in st.session_state:
        df_meta = st.session_state.get("pipeline_meta", {})
        st.success(f"✅ {df_meta.get('final_count', 0)} transactions loaded")
        if st.button("🗑️ Clear & Upload New", use_container_width=True):
            keys_to_clear = [
                k for k in st.session_state.keys()
                if k in ["transactions", "pipeline_meta"] or k.startswith("processed_")
            ]
            for key in keys_to_clear:
                del st.session_state[key]
            st.rerun()
        st.markdown("---")

    st.markdown("#### 🧠 Classifier")
    if st.button("🔄 Retrain Model", use_container_width=True):
        with st.spinner("Training..."):
            try:
                from classifier.feedback import retrain_with_feedback
                metrics = retrain_with_feedback()
                st.success(
                    f"Done! F1: {metrics['cv_f1_mean']:.3f} "
                    f"({metrics.get('user_corrections', 0)} corrections included)"
                )
            except Exception as e:
                st.error(f"Failed: {e}")

    st.markdown("---")
    st.markdown("#### ℹ️ How it works")
    st.markdown(
        """
        1. **Upload** your bank PDF
        2. **Extract** — pdfplumber reads the transaction table
        3. **Parse** — CRF labels tokens to extract merchant names
        4. **Classify** — Character TF-IDF + LinearSVC assigns categories
        5. **Visualize** — charts and insights

        Supports **HDFC · SBI · ICICI · CUB · Axis · Kotak** and more.

        🔒 **100% offline** — no data leaves your machine.
        """
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("💰 Financial Transaction Analyzer")
st.markdown(
    "Upload your bank statement PDF to automatically extract, classify, and visualize your spending."
)

# ---------------------------------------------------------------------------
# File Upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Bank Statement PDF",
    type=["pdf"],
    help="Supports HDFC, SBI, ICICI, and other bank formats.",
)

# Check if we have demo data or uploaded data already processed
if uploaded_file is None and "transactions" not in st.session_state:
    st.info(
        "👆 Upload a bank statement PDF to get started. "
        "The system will extract and classify your transactions automatically."
    )

    # Show demo data option
    st.markdown("---")
    st.subheader("🎯 Try with Demo Data")
    if st.button("Load Demo Transactions", use_container_width=True):
        # Generate synthetic demo data
        import numpy as np

        np.random.seed(42)
        demo_data = {
            "date": pd.date_range("2024-01-01", periods=50, freq="3D").strftime("%Y-%m-%d").tolist(),
            "description": [
                "SWIGGY ORDER PAYMENT", "AMAZON ONLINE PURCHASE", "IRCTC TRAIN TICKET",
                "ELECTRICITY BILL PAYMENT", "BOOKMYSHOW MOVIE TICKET", "ZOMATO FOOD DELIVERY",
                "UBER RIDE PAYMENT", "NETFLIX SUBSCRIPTION", "FLIPKART ORDER PAYMENT",
                "SALARY CREDIT", "DOMINOS PIZZA ONLINE", "AIRTEL POSTPAID BILL",
                "MAKEMYTRIP FLIGHT BOOKING", "STEAM GAME PURCHASE", "MYNTRA FASHION PURCHASE",
                "STARBUCKS COFFEE", "OLA CABS RIDE", "AMAZON PRIME SUBSCRIPTION",
                "PVR CINEMAS TICKET", "BIG BASKET ONLINE GROCERY", "NYKAA BEAUTY PRODUCTS",
                "PETROL PUMP FUEL PURCHASE", "SPOTIFY PREMIUM SUBSCRIPTION", "CONCERT TICKET BOOKING",
                "NEFT TRANSFER RECEIVED", "ATM CASH WITHDRAWAL", "HOSPITAL BILL PAYMENT",
                "ZEPTO GROCERY ORDER", "RAPIDO BIKE TAXI", "GAMING ZONE PAYMENT",
                "KFC OUTLET PAYMENT", "RELIANCE DIGITAL PURCHASE", "OYO ROOMS BOOKING",
                "WATER BILL PAYMENT", "INOX MOVIES TICKET", "CAFE COFFEE DAY",
                "LYFT RIDE PAYMENT", "HOTSTAR SUBSCRIPTION", "AJIO CLOTHING PURCHASE",
                "MUTUAL FUND PURCHASE SIP", "PIZZA HUT ORDER", "CROMA ELECTRONICS STORE",
                "GOIBIBO HOTEL BOOKING", "GAS BILL PAYMENT", "PLAYSTATION STORE PURCHASE",
                "BLINKIT QUICK COMMERCE", "DECATHLON SPORTS GOODS", "INDIGO AIRLINES TICKET",
                "LIC PREMIUM PAYMENT", "UDEMY COURSE PURCHASE",
            ],
            "amount": [
                -450, -2399, -1250, -1800, -600, -380, -220, -649, -1599, 85000,
                -520, -999, -5500, -1299, -2100, -350, -180, -1499, -450, -1200,
                -899, -2500, -119, -2000, 10000, -5000, -3500, -650, -80, -500,
                -320, -4500, -2800, -350, -380, -280, -190, -299, -1800, -5000,
                -480, -8999, -6200, -420, -999, -720, -3200, -7500, -2400, -499,
            ],
            "balance": None,
        }
        demo_df = pd.DataFrame(demo_data)
        demo_df["balance"] = 50000 + demo_df["amount"].cumsum()

        from classifier.predict import TransactionClassifier
        clf = TransactionClassifier()
        demo_df = clf.classify_dataframe(demo_df)
        st.session_state["transactions"] = demo_df
        st.session_state["pipeline_meta"] = {
            "raw_count": 50,
            "final_count": 50,
            "extraction_method": "demo",
            "page_count": 0,
        }
        st.rerun()

    st.stop()

# ---------------------------------------------------------------------------
# Process uploaded PDF
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    process_key = f"processed_{uploaded_file.name}_{uploaded_file.size}"

    if process_key not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner(
            f"⚡ Processing '{uploaded_file.name}'... Extracting and classifying transactions."
        ):
            progress = st.progress(0, text="Extracting tables from PDF...")
            try:
                pipeline = FinancialPipeline()
                progress.progress(20, text="Parsing transaction table...")
                result = pipeline.process(tmp_path)
                progress.progress(90, text="Classifying transactions...")
                progress.progress(100, text="Done!")

                # Only cache if we actually got transactions
                if result["final_count"] > 0:
                    st.session_state[process_key] = result
                st.session_state["transactions"] = result["transactions"]
                st.session_state["pipeline_meta"] = {
                    "raw_count": result["raw_count"],
                    "final_count": result["final_count"],
                    "extraction_method": result["extraction_method"],
                    "page_count": result["page_count"],
                }

                if result["errors"]:
                    for err in result["errors"]:
                        st.warning(f"⚠️ {err}")

            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")
                logger.exception("Pipeline error")
                st.stop()
            finally:
                os.unlink(tmp_path)

df: pd.DataFrame = st.session_state.get("transactions", pd.DataFrame())
meta = st.session_state.get("pipeline_meta", {})

if df.empty:
    st.error("No transactions were extracted. Please check that the PDF is a valid digital bank statement.")
    st.stop()

# ---------------------------------------------------------------------------
# Summary Metrics
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Summary")

total_debit = df[df["amount"] < 0]["amount"].abs().sum()
total_credit = df[df["amount"] > 0]["amount"].sum()
net = total_credit - total_debit
n_txns = len(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{n_txns:,}")
col2.metric("Total Spend", f"₹{total_debit:,.2f}")
col3.metric("Total Income", f"₹{total_credit:,.2f}")
col4.metric("Net Balance Change", f"₹{net:,.2f}", delta=f"{'▲' if net >= 0 else '▼'}")

if meta:
    st.caption(
        f"Extraction: {meta.get('extraction_method', 'N/A')} | "
        f"Pages: {meta.get('page_count', 'N/A')} | "
        f"Raw rows parsed: {meta.get('raw_count', 'N/A')} | "
        f"After processing: {meta.get('final_count', 'N/A')}"
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📋 Transactions", "🥧 Category Breakdown", "📅 Monthly Trends", "🏪 Top Merchants", "💡 Insights", "🎯 Feedback & Training"]
)

# ── Tab 1: Transactions Table ─────────────────────────────────────────────
with tab1:
    st.subheader("All Transactions")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        cat_filter = st.multiselect(
            "Filter by Category",
            options=sorted(df["category"].unique()) if "category" in df.columns else [],
            default=[],
        )
    with col_f2:
        txn_type = st.radio("Transaction Type", ["All", "Debits", "Credits"], horizontal=True)
    with col_f3:
        search = st.text_input("Search Description", placeholder="e.g. SWIGGY")

    filtered = df.copy()
    if cat_filter:
        filtered = filtered[filtered["category"].isin(cat_filter)]
    if txn_type == "Debits":
        filtered = filtered[filtered["amount"] < 0]
    elif txn_type == "Credits":
        filtered = filtered[filtered["amount"] > 0]
    if search:
        filtered = filtered[
            filtered["description"].str.contains(search, case=False, na=False)
        ]

    # Display
    display_df = filtered.copy()
    # Drop internal columns not useful for display
    if "raw_description" in display_df.columns:
        display_df = display_df.drop(columns=["raw_description"])
    display_df["amount"] = display_df["amount"].apply(
        lambda x: f"₹{x:,.2f}" if x >= 0 else f"-₹{abs(x):,.2f}"
    )
    if "balance" in display_df.columns:
        display_df["balance"] = display_df["balance"].apply(
            lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—"
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        height=450,
        column_config={
            "date": st.column_config.TextColumn("Date"),
            "description": st.column_config.TextColumn("Description", width="large"),
            "amount": st.column_config.TextColumn("Amount"),
            "balance": st.column_config.TextColumn("Balance"),
            "category": st.column_config.TextColumn("Category"),
        },
    )
    st.caption(f"Showing {len(filtered):,} of {len(df):,} transactions")

    # Download
    export_df = filtered.drop(columns=["raw_description"], errors="ignore")
    csv = export_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download as CSV",
        data=csv,
        file_name="transactions.csv",
        mime="text/csv",
    )

# ── Tab 2: Category Breakdown ─────────────────────────────────────────────
with tab2:
    st.subheader("Spending by Category")

    cat_summary = compute_category_summary(df)

    if cat_summary.empty:
        st.info("No debit transactions found.")
    else:
        col_pie, col_bar = st.columns(2)

        with col_pie:
            fig_pie = px.pie(
                cat_summary,
                values="total_spent",
                names="category",
                title="Spend Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            fig_bar = px.bar(
                cat_summary,
                x="category",
                y="total_spent",
                title="Total Spend per Category",
                color="category",
                color_discrete_sequence=px.colors.qualitative.Set3,
                text="total_spent",
            )
            fig_bar.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
            fig_bar.update_layout(
                showlegend=False,
                height=400,
                yaxis_title="Amount (₹)",
                xaxis_title="Category",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Category detail table
        st.subheader("Category Details")
        detail_df = cat_summary.copy()
        detail_df["total_spent"] = detail_df["total_spent"].apply(lambda x: f"₹{x:,.2f}")
        detail_df["avg_transaction"] = detail_df["avg_transaction"].apply(lambda x: f"₹{x:,.2f}")
        detail_df.columns = ["Category", "Total Spent", "# Transactions", "Avg Transaction"]
        st.dataframe(detail_df, use_container_width=True, hide_index=True, height=280)

# ── Tab 3: Monthly Trends ─────────────────────────────────────────────────
with tab3:
    st.subheader("Monthly Spending Trends")

    monthly_summary = compute_monthly_summary(df)
    monthly_totals = compute_monthly_totals(df)

    if monthly_summary.empty:
        st.info("Not enough data for monthly analysis.")
    else:
        # Stacked bar by category
        fig_monthly = px.bar(
            monthly_summary,
            x="month",
            y="total_spent",
            color="category",
            title="Monthly Spend by Category",
            barmode="stack",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_monthly.update_layout(
            height=400,
            yaxis_title="Amount (₹)",
            xaxis_title="Month",
            legend_title="Category",
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Income vs Expense
        if not monthly_totals.empty:
            fig_ie = go.Figure()
            fig_ie.add_trace(
                go.Bar(
                    x=monthly_totals["month"],
                    y=monthly_totals["total_credit"],
                    name="Income",
                    marker_color="#2ecc71",
                )
            )
            fig_ie.add_trace(
                go.Bar(
                    x=monthly_totals["month"],
                    y=monthly_totals["total_debit"],
                    name="Expense",
                    marker_color="#e74c3c",
                )
            )
            fig_ie.add_trace(
                go.Scatter(
                    x=monthly_totals["month"],
                    y=monthly_totals["net"],
                    name="Net",
                    mode="lines+markers",
                    line=dict(color="#3498db", width=2),
                    marker=dict(size=8),
                )
            )
            fig_ie.update_layout(
                title="Monthly Income vs Expense",
                barmode="group",
                height=350,
                yaxis_title="Amount (₹)",
                xaxis_title="Month",
            )
            st.plotly_chart(fig_ie, use_container_width=True)

# ── Tab 4: Top Merchants ──────────────────────────────────────────────────
with tab4:
    st.subheader("Top Merchants by Spend")

    n_merchants = st.slider("Show top N merchants", 5, 20, 10)
    top_merchants = get_top_merchants(df, n=n_merchants)

    if top_merchants.empty:
        st.info("No merchant data available.")
    else:
        fig_merch = px.bar(
            top_merchants,
            x="total_spent",
            y="description",
            orientation="h",
            title=f"Top {n_merchants} Merchants",
            color="total_spent",
            color_continuous_scale="Blues",
            text="total_spent",
        )
        fig_merch.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        fig_merch.update_layout(
            height=max(300, n_merchants * 40),
            yaxis=dict(autorange="reversed"),
            xaxis_title="Total Spent (₹)",
            yaxis_title="",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_merch, use_container_width=True)

        # Table
        display_merch = top_merchants.copy()
        display_merch["total_spent"] = display_merch["total_spent"].apply(lambda x: f"₹{x:,.2f}")
        display_merch.columns = ["Merchant", "Total Spent", "Transactions"]
        st.dataframe(display_merch, use_container_width=True, hide_index=True, height=450)

# ── Tab 5: Insights ───────────────────────────────────────────────────────
with tab5:
    st.subheader("💡 Spending Insights")

    insights = generate_insights(df)
    for insight in insights:
        st.info(f"💡 {insight}")

    st.markdown("---")
    st.subheader("Category Spending Heatmap")

    monthly_cat = compute_monthly_summary(df)
    if not monthly_cat.empty:
        pivot = monthly_cat.pivot(index="category", columns="month", values="total_spent").fillna(0)
        fig_heat = px.imshow(
            pivot,
            title="Spend Heatmap (Category × Month)",
            color_continuous_scale="Blues",
            aspect="auto",
            text_auto=".0f",
        )
        fig_heat.update_layout(height=350)
        st.plotly_chart(fig_heat, use_container_width=True)


# ── Tab 6: Feedback & Training ────────────────────────────────────────────
with tab6:
    from classifier.feedback import (
        load_corrections,
        save_correction,
        save_corrections_batch,
        load_user_merchants,
        save_merchant,
        delete_merchant,
        retrain_with_feedback,
        get_feedback_stats,
        clear_corrections,
        clear_merchants,
    )
    from classifier.train import CATEGORIES

    st.subheader("🎯 Improve the Model")
    st.markdown(
        "Correct misclassified transactions and add new merchants. "
        "Your feedback is saved locally and used to retrain the model."
    )

    # ── Section 1: Fix Transaction Categories ─────────────────────────────
    st.markdown("---")
    st.markdown("### 📝 Correct Transaction Categories")
    st.caption(
        "Select transactions that were misclassified and assign the correct category. "
        "After making corrections, click 'Retrain Model' to update the classifier."
    )

    # Show editable transaction table
    edit_df = df[["date", "description", "amount", "category"]].copy()
    edit_df = edit_df.reset_index(drop=True)
    edit_df["correct_category"] = edit_df["category"]

    edited = st.data_editor(
        edit_df,
        column_config={
            "date": st.column_config.TextColumn("Date", disabled=True),
            "description": st.column_config.TextColumn("Description", disabled=True, width="large"),
            "amount": st.column_config.NumberColumn("Amount", disabled=True, format="₹%.2f"),
            "category": st.column_config.TextColumn("Current", disabled=True),
            "correct_category": st.column_config.SelectboxColumn(
                "Correct Category",
                options=CATEGORIES,
                required=True,
            ),
        },
        use_container_width=True,
        height=350,
        hide_index=True,
        key="category_editor",
    )

    # Find rows where user changed the category
    if edited is not None:
        changed_mask = edited["correct_category"] != edited["category"]
        changed_rows = edited[changed_mask]

        if not changed_rows.empty:
            st.info(f"🔄 {len(changed_rows)} correction(s) pending")

            if st.button("💾 Save Corrections", type="primary", use_container_width=True):
                corrections_to_save = []
                for _, row in changed_rows.iterrows():
                    # Find raw_description from the original df
                    raw = ""
                    if "raw_description" in df.columns:
                        match = df[df["description"] == row["description"]]
                        if not match.empty:
                            raw = match.iloc[0].get("raw_description", "")

                    corrections_to_save.append({
                        "description": row["description"],
                        "raw_description": raw,
                        "category": row["correct_category"],
                        "original_category": row["category"],
                    })

                count = save_corrections_batch(corrections_to_save)
                st.success(f"✅ Saved {count} correction(s)")

    # ── Section 2: Add New Merchants ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏪 Add New Merchant")
    st.caption(
        "Map a merchant name fragment to a category. "
        "This works instantly (no retrain needed) — the lookup table is checked first."
    )

    col_m1, col_m2, col_m3 = st.columns([2, 2, 1])
    with col_m1:
        new_fragment = st.text_input(
            "Merchant fragment (substring to match)",
            placeholder="e.g. 'brindha' matches 'BRINDHAC'",
            key="new_merchant_fragment",
        )
    with col_m2:
        new_merchant_name = st.text_input(
            "Clean display name",
            placeholder="e.g. 'Brindha Cafe'",
            key="new_merchant_name",
        )
    with col_m3:
        new_merchant_cat = st.selectbox(
            "Category",
            options=CATEGORIES,
            key="new_merchant_category",
        )

    if st.button("➕ Add Merchant", use_container_width=True):
        if new_fragment and new_merchant_name:
            save_merchant(new_fragment, new_merchant_name, new_merchant_cat)
            st.success(f"✅ Added: '{new_fragment}' → {new_merchant_name} ({new_merchant_cat})")
            st.caption("This mapping is active immediately. No retrain needed.")
        else:
            st.warning("Please fill in both fragment and display name.")

    # Show existing user merchants
    user_merchants = load_user_merchants()
    if user_merchants:
        st.markdown("**Your custom merchants:**")
        merch_data = [
            {"Fragment": k, "Name": v[0], "Category": v[1]}
            for k, v in sorted(user_merchants.items())
        ]
        st.dataframe(
            pd.DataFrame(merch_data),
            use_container_width=True,
            hide_index=True,
            height=200,
        )

    # ── Section 3: Retrain ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧠 Retrain Classifier")

    stats = get_feedback_stats()
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Saved Corrections", stats["total_corrections"])
    col_s2.metric("Custom Merchants", stats["total_merchants"])

    if stats["corrections_by_category"]:
        st.caption(f"Corrections by category: {stats['corrections_by_category']}")

    st.markdown(
        "Retraining merges your corrections with the base training data "
        "and fits a new model. Custom merchants work instantly without retraining."
    )

    if st.button("🚀 Retrain with Feedback", type="primary", use_container_width=True):
        with st.spinner("Training classifier with your corrections..."):
            try:
                metrics = retrain_with_feedback()
                st.success(
                    f"✅ Model retrained! "
                    f"CV F1: {metrics['cv_f1_mean']:.3f} | "
                    f"Samples: {metrics['n_samples']} "
                    f"(+{metrics.get('user_corrections', 0)} from feedback)"
                )
                st.info("Re-upload your PDF or refresh to see updated classifications.")
            except Exception as e:
                st.error(f"❌ Training failed: {e}")

    # ── Section 4: Export / Import Feedback ───────────────────────────────
    st.markdown("---")
    st.markdown("### 📤 Export / Manage Feedback Data")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        corrections = load_corrections()
        if corrections:
            corrections_json = json.dumps(corrections, indent=2)
            st.download_button(
                "⬇️ Export Corrections (JSON)",
                data=corrections_json,
                file_name="feedback_corrections.json",
                mime="application/json",
            )
        else:
            st.caption("No corrections saved yet.")

    with col_e2:
        if user_merchants:
            merchants_json = json.dumps(
                {k: list(v) for k, v in user_merchants.items()}, indent=2
            )
            st.download_button(
                "⬇️ Export Merchants (JSON)",
                data=merchants_json,
                file_name="feedback_merchants.json",
                mime="application/json",
            )
        else:
            st.caption("No custom merchants saved yet.")

    # Import feedback
    st.markdown("**Import feedback from JSON:**")
    uploaded_feedback = st.file_uploader(
        "Upload corrections or merchants JSON",
        type=["json"],
        key="feedback_upload",
    )
    if uploaded_feedback is not None:
        try:
            import json as json_mod
            data = json_mod.loads(uploaded_feedback.read())

            if isinstance(data, list) and data and "description" in data[0]:
                # It's corrections
                count = save_corrections_batch(data)
                st.success(f"✅ Imported {count} corrections")
            elif isinstance(data, dict):
                # It's merchants
                for frag, (name, cat) in data.items():
                    save_merchant(frag, name, cat)
                st.success(f"✅ Imported {len(data)} merchants")
            else:
                st.warning("Unrecognized JSON format.")
        except Exception as e:
            st.error(f"Failed to import: {e}")

    # Clear buttons
    with st.expander("⚠️ Danger Zone"):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if st.button("🗑️ Clear All Corrections"):
                clear_corrections()
                st.success("Corrections cleared.")
        with col_d2:
            if st.button("🗑️ Clear All Custom Merchants"):
                clear_merchants()
                st.success("Custom merchants cleared.")
