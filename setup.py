from setuptools import setup, find_packages

setup(
    name="projectowl",
    version="0.1.0",
    description="Time-series characterization and stock price prediction",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "sqlalchemy>=2.0",
        "psycopg2-binary>=2.9",
        "python-dotenv>=1.0",
        "requests>=2.31",
        "Nasdaq-Data-Link>=1.0",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "plotly>=5.15",
        "dash>=2.14",
        "arch>=6.0",
        "statsmodels>=0.14",
        "tqdm>=4.65",
    ],
)
