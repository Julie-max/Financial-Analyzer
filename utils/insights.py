"""
Insights Engine
Generates spending summaries, monthly aggregations, and natural-language insights.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


def compute_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total spend per category (debits only).

    Returns:
        DataFrame with columns: category, total_spent, transaction_count, avg_transaction
    """
    if df.empty or "category" not in df.columns:
        return pd.DataFrame(
            columns=["category", "total_spent", "transaction_count", "avg_transaction"]
        )

    debits = df[df["amount"] < 0].copy()
    debits["abs_amount"] = debits["amount"].abs()

    summary = (
        debits.groupby("category")["abs_amount"]
        .agg(total_spent="sum", transaction_count="count", avg_transaction="mean")
        .reset_index()
        .sort_values("total_spent", ascending=False)
    )
    return summary


def compute_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly spend per category.

    Returns:
        DataFrame with columns: month, category, total_spent
    """
    if df.empty or "category" not in df.columns:
        return pd.DataFrame(columns=["month", "category", "total_spent"])

    debits = df[df["amount"] < 0].copy()
    debits["abs_amount"] = debits["amount"].abs()
    debits["month"] = pd.to_datetime(debits["date"]).dt.to_period("M").astype(str)

    monthly = (
        debits.groupby(["month", "category"])["abs_amount"]
        .sum()
        .reset_index()
        .rename(columns={"abs_amount": "total_spent"})
        .sort_values(["month", "total_spent"], ascending=[True, False])
    )
    return monthly


def compute_monthly_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total debit and credit per month.

    Returns:
        DataFrame with columns: month, total_debit, total_credit, net
    """
    if df.empty:
        return pd.DataFrame(columns=["month", "total_debit", "total_credit", "net"])

    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

    debits = df[df["amount"] < 0].groupby("month")["amount"].sum().abs().rename("total_debit")
    credits = df[df["amount"] > 0].groupby("month")["amount"].sum().rename("total_credit")

    monthly = pd.concat([debits, credits], axis=1).fillna(0).reset_index()
    monthly["net"] = monthly["total_credit"] - monthly["total_debit"]
    return monthly.sort_values("month")


def generate_insights(df: pd.DataFrame) -> List[str]:
    """
    Generate natural-language spending insights.

    Returns:
        List of insight strings.
    """
    insights = []

    if df.empty or "category" not in df.columns:
        return ["No transaction data available for insights."]

    debits = df[df["amount"] < 0].copy()
    debits["abs_amount"] = debits["amount"].abs()

    if debits.empty:
        return ["No debit transactions found."]

    total_spend = debits["abs_amount"].sum()
    insights.append(f"Total spend: ₹{total_spend:,.2f}")

    # Top category
    cat_summary = compute_category_summary(df)
    if not cat_summary.empty:
        top_cat = cat_summary.iloc[0]
        pct = (top_cat["total_spent"] / total_spend) * 100
        insights.append(
            f"Highest spending category: {top_cat['category']} "
            f"(₹{top_cat['total_spent']:,.2f}, {pct:.1f}% of total)"
        )

    # Monthly comparison
    debits["month"] = pd.to_datetime(debits["date"]).dt.to_period("M").astype(str)
    monthly_totals = debits.groupby("month")["abs_amount"].sum()

    if len(monthly_totals) >= 2:
        months = monthly_totals.index.tolist()
        last_month = monthly_totals.iloc[-1]
        prev_month = monthly_totals.iloc[-2]
        change_pct = ((last_month - prev_month) / prev_month) * 100 if prev_month > 0 else 0

        direction = "more" if change_pct > 0 else "less"
        insights.append(
            f"You spent {abs(change_pct):.1f}% {direction} in {months[-1]} "
            f"compared to {months[-2]} "
            f"(₹{last_month:,.2f} vs ₹{prev_month:,.2f})"
        )

        # Per-category monthly comparison
        if "category" in debits.columns:
            cat_monthly = debits.groupby(["month", "category"])["abs_amount"].sum().unstack(fill_value=0)
            if len(cat_monthly) >= 2:
                for cat in cat_monthly.columns:
                    last = cat_monthly[cat].iloc[-1]
                    prev = cat_monthly[cat].iloc[-2]
                    if prev > 0 and last > 0:
                        cat_change = ((last - prev) / prev) * 100
                        if abs(cat_change) >= 20:
                            direction = "more" if cat_change > 0 else "less"
                            insights.append(
                                f"You spent {abs(cat_change):.1f}% {direction} on {cat} "
                                f"in {months[-1]} vs {months[-2]}"
                            )

    # Largest single transaction
    largest = debits.loc[debits["abs_amount"].idxmax()]
    insights.append(
        f"Largest transaction: ₹{largest['abs_amount']:,.2f} — "
        f"{largest['description']} on {largest['date']}"
    )

    # Average daily spend
    date_range = (
        pd.to_datetime(debits["date"]).max() - pd.to_datetime(debits["date"]).min()
    ).days + 1
    if date_range > 0:
        avg_daily = total_spend / date_range
        insights.append(f"Average daily spend: ₹{avg_daily:,.2f}")

    return insights


def get_top_merchants(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top N merchants by total spend."""
    if df.empty:
        return pd.DataFrame(columns=["description", "total_spent", "count"])

    debits = df[df["amount"] < 0].copy()
    debits["abs_amount"] = debits["amount"].abs()

    top = (
        debits.groupby("description")["abs_amount"]
        .agg(total_spent="sum", count="count")
        .reset_index()
        .sort_values("total_spent", ascending=False)
        .head(n)
    )
    return top
