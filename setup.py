from pathlib import Path

import setuptools

VERSION = "0.1.0"

NAME = "xai-agg"

INSTALL_REQUIRES = [
    "scikit-learn",
    "numpy",
    "pandas",
    "lime",
    "shap",
    "alibi",
    "scipy",
    "tensorflow-cpu",
    "tf-keras",
    "pathos",
    "pymcdm",
    "ranx",
    "xlrd",
    "pandera",
    "ipython"
]


setuptools.setup(
    name=NAME,
    version=VERSION,
    description="Rank based, multi-criteria aggregation method for explainable AI models.",
    url="https://github.com/hiaac-finance/xai_aggregation",
    project_urls={
        "Source Code": "https://github.com/hiaac-finance/xai_aggregation",
        "Documentation": "https://xai-agg.readthedocs.io/en/latest/"
    },
    author="Everton Colombo",
    author_email="everton.colombo@students.ic.unicamp.br",
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    packages=["xai_agg"],
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
)