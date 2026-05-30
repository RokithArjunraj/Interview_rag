"""
config.py — central settings for the entire project.
"""

import os
import re
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data" / "raw"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "interview_chunks"

# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
LLM_MODEL      = "llama-3.1-8b-instant"
LLM_MAX_TOKENS = 2000

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K             = 30   # fetch more, re-rank by recency after
RECENCY_BONUS     = 0.05 # score boost per batch above batch 1 (e.g. batch 9 → +0.40)
RECENT_BATCH_PREF = 2    # extra weight multiplier for last N batches

# ── Batch number extraction ───────────────────────────────────────────────────
def extract_batch_from_filename(filename: str) -> int:
    """
    Auto-detects batch number from PDF filename.
    Works for patterns like:
      Batch_10_, batch10, Batch 9, _B9_, etc.
    Falls back to 1 if not found.
    """
    m = re.search(r'[Bb]atch[\s_\-]*(\d+)', filename)
    if m:
        return int(m.group(1))
    # fallback: any standalone number in filename
    nums = re.findall(r'\d+', filename)
    if nums:
        return int(nums[0])
    return 1

# ── Topic taxonomy ────────────────────────────────────────────────────────────
TOPIC_MAP = {
    "linear_regression":  ["linear regression", "multicollinearity", "vif", "r square",
                            "adj r2", "adjusted r", "lasso", "ridge", "regularization",
                            "regression assumptions", "residual", "heteroskedasticity",
                            "stepwise", "aic", "bic", "polynomial regression", "ols"],
    "classification":     ["logistic regression", "classification", "f1 score", "precision",
                            "recall", "roc", "auc", "confusion matrix", "type 1 error",
                            "type 2 error", "false positive", "smote", "class imbalance",
                            "threshold", "sigmoid"],
    "tree_models":        ["decision tree", "random forest", "xgboost", "gradient boosting",
                            "catboost", "lightgbm", "bagging", "boosting", "pruning"],
    "time_series":        ["time series", "arima", "sarima", "stationarity", "adf test",
                            "acf", "pacf", "ljung box", "exogenous", "forecasting",
                            "seasonality", "cointegration", "mean reverting", "pairs trading"],
    "statistics":         ["hypothesis testing", "p-value", "t-test", "chi square", "anova",
                            "central limit theorem", "clt", "confidence interval",
                            "normality", "shapiro", "kolmogorov", "a/b testing",
                            "type 1", "type 2", "proportion test"],
    "dimensionality":     ["pca", "svd", "eigenvalue", "eigenvector", "dimensionality reduction",
                            "feature selection", "variance explained"],
    "clustering":         ["clustering", "k-means", "kmeans", "dbscan", "hierarchical",
                            "silhouette", "davies bouldin", "elbow method"],
    "nlp_llm":            ["nlp", "bert", "transformer", "attention", "llm", "rag",
                            "retrieval augmented", "embeddings", "word2vec", "glove",
                            "tf-idf", "fine tuning", "gpt", "encoder", "decoder"],
    "deep_learning":      ["neural network", "deep learning", "lstm", "cnn", "mlp",
                            "vanishing gradient", "batch normalization", "dropout",
                            "backprop", "optimizer", "skip connection"],
    "python_coding":      ["python", "pandas", "numpy", "sql", "dsa", "algorithm",
                            "time complexity", "fibonacci", "prime number", "coding"],
    "probability":        ["probability", "bayes", "conditional probability", "distribution",
                            "binomial", "poisson", "normal distribution", "expectation"],
    "finance_domain":     ["finance", "trading", "commodity", "futures", "arbitrage",
                            "portfolio", "derivatives", "hedge fund", "credit risk",
                            "loan default", "fraud detection"],
    "recommendation":     ["recommendation system", "collaborative filtering",
                            "content based", "matrix factorization", "two tower model"],
    "business_case":      ["case study", "guesstimate", "business problem", "revenue",
                            "market segmentation", "churn", "pricing", "demand forecast"],
}

# ── Companies ─────────────────────────────────────────────────────────────────
# Add new companies here as you collect more batch docs
COMPANIES = [
    "QRT", "DE SHAW", "SUN PHARMA", "JPMC", "MASTER CARD", "SWIGGY",
    "PNG", "KENVUE", "BCG", "POLYCAB", "PIRAMAL FINANCE", "OLIVER WYMAN",
    "RENEW", "AUXO AI", "DECISION POINT", "L & T FINANCE", "DECIMAL POINT", "JSW",
    "AMAZON", "GOOGLE", "MICROSOFT", "FLIPKART", "MEESHO", "WALMART",
    "DELOITTE", "MCKINSEY", "BAIN", "EY", "KPMG", "PWC",
]

# ── Company name aliases ──────────────────────────────────────────────────────
# Maps messy variants found in docs → canonical name
COMPANY_ALIASES = {
    "D E SHAW":              "DE SHAW",
    "D.E.SHAW":              "DE SHAW",
    "D.E. SHAW":             "DE SHAW",
    "DE SHAW & CO":          "DE SHAW",
    "MASTERCARD AI GARAGE":  "MASTER CARD",
    "MASTERCARD":            "MASTER CARD",
    "MASTER CARD AI":        "MASTER CARD",
    "P&G":                   "PNG",
    "PROCTER & GAMBLE":      "PNG",
    "PROCTER AND GAMBLE":    "PNG",
    "PIRAMAL HOUSING FINANCE": "PIRAMAL FINANCE",
    "PLAY GAMES 24*7 PVT LTD": "GAMES24X7",
    "PLAYGAMES":             "GAMES24X7",
    "L&T FINANCE":           "L & T FINANCE",
    "LT FINANCE":            "L & T FINANCE",
    "OLIVER WYMAN":          "OLIVER WYMAN",
    "RENEW POWER":           "RENEW",
}
