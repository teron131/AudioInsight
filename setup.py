from setuptools import find_packages, setup

setup(
    name="audioinsight",
    description="Real-time, Fully Local Speech-to-Text and Speaker Diarization with AudioInsight",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Teron",
    url="https://github.com/teron131/AudioInsight",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "ffmpeg-python",
        "librosa",
        "soundfile",
        "faster-whisper",
        "uvicorn",
        "websockets",
        "numpy",
        "pydantic",
        "python-dotenv",
    ],
    extras_require={
        "diarization": [
            "diart",
            "pyannote.audio",
            "torch>=2.6.0,<2.7.0",
            "torchvision>=0.21.0,<0.22.0",
            "torchaudio>=2.6.0,<2.7.0",
            "huggingface-hub",
        ],
        "vac": ["torch>=2.6.0,<2.7.0"],
        "sentence": ["mosestokenizer"],
        "whisper": ["whisper", "opencc"],
        "openai": ["openai"],
        "llm": [
            "langchain",
            "langchain-openai",
            "langchain-google-genai",
            "python-dotenv",
            "pydantic",
        ],
        "complete": [
            # All extras combined for full functionality
            "diart",
            "pyannote.audio",
            "torch>=2.6.0,<2.7.0",
            "torchvision>=0.21.0,<0.22.0",
            "torchaudio>=2.6.0,<2.7.0",
            "huggingface-hub",
            "mosestokenizer",
            "whisper",
            "opencc",
            "openai",
            "langchain",
            "langchain-openai",
            "langchain-google-genai",
        ],
    },
    package_data={
        "audioinsight": ["frontend/*.html", "server/*.py"],
    },
    entry_points={
        "console_scripts": [
            "audioinsight-server=audioinsight.app:main",
        ],
    },
    python_requires=">=3.9",
)
