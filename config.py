"""
config.py — central settings for the entire project.
Change values here; nothing else needs to be edited for basic setup.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data" / "raw"        # drop PDFs here
CHROMA_DIR  = BASE_DIR / "data" / "chroma_db"  # vector store lives here

# ── Embedding model ───────────────────────────────────────────────────────────
# Free, runs locally, no API key needed.
# Swap to "all-mpnet-base-v2" for higher quality (slower).
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "interview_chunks"

# ── LLM (Anthropic Claude) ────────────────────────────────────────────────────
# Set your key as env variable: export ANTHROPIC_API_KEY=sk-...
# Or paste directly here (not recommended for shared code).
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
LLM_MODEL     = "llama-3.1-8b-instant"   # free, very fast on Groq
LLM_MAX_TOKENS = 1500

# ── Retrieval ─────────────────────────────────────────────────────────────────
# How many chunks to fetch per query before aggregation
TOP_K = 15

# ── Topic taxonomy ────────────────────────────────────────────────────────────
# Maps a canonical tag → keywords that trigger it during parsing.
# Add more tags/keywords as you collect more docs.
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
                            "normality", "shapiro", "kolmogorov", "a/b testing"],
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

# ── Companies list (used by parser to detect section headers) ─────────────────
COMPANIES = [
    "QRT", "DE SHAW", "SUN PHARMA", "JPMC", "MASTER CARD", "SWIGGY",
    "PNG", "KENVUE", "BCG", "POLYCAB", "PIRAMAL FINANCE", "OLIVER WYMAN",
    "RENEW", "AUXO AI", "DECISION POINT", "L & T FINANCE", "DECIMAL POINT", "JSW"
]
