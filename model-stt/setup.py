from setuptools import setup

# The container installs the ct2 extra only: faster-whisper (CTranslate2) needs no
# torch, which keeps the multi-GB torch/nvidia wheel stack out of the image.
CT2_BACKEND = [
    'faster-whisper>=1.1.0',
    # ctranslate2 links cuBLAS/cuDNN from these wheels rather than a system CUDA
    # install; without them a torch-free image cannot run on GPU
    'nvidia-cublas-cu12',
    'nvidia-cudnn-cu12>=9,<10',
]

# Reference implementation, kept for bench/ comparisons only. Never installed in
# the image: it pulls torch and roughly 5.5 GB of nvidia wheels.
OPENAI_BACKEND = [
    'openai-whisper==20250625',
    'torch',
]

# DISABLED (translation): ollama is the client for path C's LLM translation.
# It is an extra rather than a runtime dependency now that translation is out of
# scope -- restoring path C means adding it back to install_requires.
TRANSLATE = [
    'ollama',
]

BENCH = [
    'jiwer',
    'sacrebleu',
    'datasets',
]

setup(
    name="model-stt",
    version="0.1",
    packages=['src'],
    install_requires=[
        'common-ml @ git+https://github.com/eluv-io/common-ml@vector-tags',
        'ffmpeg-python==0.2.0',
        'loguru',
        'PyYAML',
        'dacite',
        'setproctitle',
    ],
    extras_require={
        'ct2': CT2_BACKEND,
        'openai': OPENAI_BACKEND,
        'translate': TRANSLATE,
        'bench': BENCH,
        'all': CT2_BACKEND + OPENAI_BACKEND + TRANSLATE + BENCH,
    },
)
