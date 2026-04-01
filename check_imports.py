mods = [
    "fastapi",
    "uvicorn",
    "pandas",
    "numpy",
    "joblib",
    "xgboost",
    "sklearn",
    "imblearn",
    "shap",
    "dotenv",
    "pyarrow",
    "langchain_core",
    "langchain_groq",
]
for m in mods:
    __import__(m)
print("core imports ok")