# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tabnav",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered browser tab organization and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tabnav",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.1",
        "openai>=1.3.3",
        "playwright>=1.40.0",
        "beautifulsoup4>=4.12.2",
        "sentence-transformers>=2.2.2",
        "umap-learn>=0.5.4",
        "hdbscan>=0.8.33",
        "numpy>=1.26.2",
        "pandas>=2.1.3",
        "pytest>=7.4.3",
        "httpx>=0.25.2",
        "pytest-asyncio>=0.21.1",
        "fasttext>=0.9.2",
        "deep-translator>=1.11.4",
        "redis>=5.0.1",
        "celery>=5.3.4",
        "prometheus-client>=0.17.1",
        "sentry-sdk>=1.32.0",
        "structlog>=23.1.0"
    ],
    extras_require={
        'dev': [
            'black',
            'isort',
            'flake8',
            'mypy',
            'pytest-cov',
            'pre-commit'
        ],
        'test': [
            'pytest',
            'pytest-asyncio',
            'pytest-cov',
            'aiohttp',
            'asynctest'
        ]
    },
    entry_points={
        'console_scripts': [
            'tabnav=tabnav.cli:main',
        ],
    },
)