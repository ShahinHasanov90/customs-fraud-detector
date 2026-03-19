"""Setup configuration for customs-fraud-detector."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="customs-fraud-detector",
    version="0.3.1",
    author="Shahin Hasanov",
    author_email="shahin.hasanov@example.com",
    description="ML-powered fraud detection for customs trade declarations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShahinHasanov90/customs-fraud-detector",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.7",
            "pre-commit>=3.5",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "fraud-detector=fraud_detector.api.app:main",
        ],
    },
)
