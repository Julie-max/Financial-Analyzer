from .insights import (
    compute_category_summary,
    compute_monthly_summary,
    compute_monthly_totals,
    generate_insights,
    get_top_merchants,
)

# FinancialPipeline imported lazily to keep startup fast
def get_pipeline(*args, **kwargs):
    from .pipeline import FinancialPipeline
    return FinancialPipeline(*args, **kwargs)

__all__ = [
    "get_pipeline",
    "compute_category_summary",
    "compute_monthly_summary",
    "compute_monthly_totals",
    "generate_insights",
    "get_top_merchants",
]
