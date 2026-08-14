from setuptools import setup

# openai-whisper needs torch; faster-whisper (CTranslate2) does not. Keeping them
# as extras lets a production image install only the CT2 runtime and skip the
# multi-GB torch/nvidia wheel stack entirely.
OPENAI_BACKEND = [
    'openai-whisper==20250625',
    'torch',
]

CT2_BACKEND = [
    'faster-whisper>=1.1.0',
    # ctranslate2 links cuBLAS/cuDNN from these wheels rather than a system CUDA
    # install; without them a torch-free image cannot run on GPU
    'nvidia-cublas-cu12',
    'nvidia-cudnn-cu12>=9,<10',
]

BENCH = [
    'jiwer',
    'sacrebleu',
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
        'ollama',
    ],
    extras_require={
        'openai': OPENAI_BACKEND,
        'ct2': CT2_BACKEND,
        'bench': BENCH,
        'all': OPENAI_BACKEND + CT2_BACKEND + BENCH,
    },
)
