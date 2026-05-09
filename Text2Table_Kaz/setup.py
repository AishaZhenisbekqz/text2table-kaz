from setuptools import setup, find_packages

setup(
    name="text2table_kaz",
    version="1.0.0",
    description="Dual-Regime Text-to-Table Generation for Kazakh (Ospan et al., IEEE Access 2024)",
    author="Assel Ospan, Madina Mansurova, Aisha Sailau, Talshyn Sarsembayeva, Amir Mosavi",
    license="Apache-2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "transformers>=4.40.0",
        "peft>=0.11.0",
        "sentence-transformers>=2.7.0",
        "scikit-learn>=1.4.2",
        "numpy>=1.26.4",
        "pyyaml>=6.0.1",
    ],
)
