from setuptools import find_packages, setup

setup(
    name="audioinsight",
    version="0.1",
    description="Real-time, Fully Local Speech-to-Text and Speaker Diarization with AudioInsight",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Teron",
    url="https://github.com/teron131/Whisper-Realtime",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "ffmpeg-python",
        "librosa",
        "soundfile",
        "faster-whisper",
        "uvicorn",
        "websockets",
    ],
    extras_require={
        "diarization": ["diart"],
        "vac": ["torch"],
        "sentence": ["mosestokenizer"],
        "whisper": ["whisper"],
        "openai": ["openai"],
        "llm": ["langchain", "langchain-openai", "python-dotenv"],
    },
    package_data={
        "audioinsight": ["web/*.html"],
    },
    entry_points={
        "console_scripts": [
            "audioinsight-server=audioinsight.server:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.9",
)
