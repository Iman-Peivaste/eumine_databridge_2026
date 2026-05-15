from setuptools import setup, find_packages

setup(
    name="eumine_databridge",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.12",
    install_requires=[
        "pymatgen",
        "mp-api",
        "jarvis-tools",
        "alignn",
        "mace-torch",
        "torch",
        "torch-geometric",
        "scikit-learn",
        "xgboost",
        "numpy",
        "pandas",
        "python-dotenv",
        "wandb",
        "optuna",
    ],
)
