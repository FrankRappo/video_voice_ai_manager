from setuptools import setup, find_packages

setup(
    name="video-voice-ai-manager",
    version="0.1.0",
    description="Universal AI-powered video & voice analyzer",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="VVAM Team",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "google-genai>=1.0.0",
    ],
    extras_require={
        "openai": ["openai"],
        "whisper": ["faster-whisper"],
        "web": ["fastapi>=0.100.0", "uvicorn>=0.20.0", "python-multipart>=0.0.5", "aiofiles>=23.0.0"],
        "download": ["yt-dlp"],
        "all": [
            "openai",
            "faster-whisper",
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
            "python-multipart>=0.0.5",
            "aiofiles>=23.0.0",
            "yt-dlp",
            "sounddevice",
        ],
    },
    entry_points={
        "console_scripts": [
            "vvam=cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "web": ["templates/*.html", "static/**/*"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
