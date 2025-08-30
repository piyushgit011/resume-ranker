from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="advanced-resume-ranker",
    version="2.0.0",
    author="Resume Ranker Team",
    author_email="team@resumeranker.com",
    description="Advanced Resume Ranking System with AI Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/advanced-resume-ranker",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Human Resources",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "resume-ranker=src.ui.main_app:main",
            "resume-ranker-api=run_api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml", "*.json"],
    },
)