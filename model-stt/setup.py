from setuptools import setup

# The container installs the ct2 extra only. CTranslate2 itself declares no CUDA
# dependencies -- it dlopens libcublas.so.12 / libcudnn.so.9 off LD_LIBRARY_PATH,
# which the Containerfile points at site-packages/nvidia/*/lib -- so these two
# wheels have to be requested explicitly.
#
# They are the same wheels torch depends on (torch 2.8+cu128 pins
# nvidia-cublas-cu12==12.8.4.1 and nvidia-cudnn-cu12==9.10.2.21, which satisfy
# both lines below), so with PUNCTUATION installed CTranslate2 runs on torch's
# copies and nothing is duplicated. Verified by running a CT2 decode and the
# punctuation model on the same GPU in one process.
CT2_BACKEND = [
    'faster-whisper>=1.1.0',
    'nvidia-cublas-cu12',
    'nvidia-cudnn-cu12>=9,<10',
]

# Punctuation restoration over the word stream (src/punctuate.py). In
# install_requires rather than an extra: whisper's punctuation is unreliable
# enough that the sentence track is not trustworthy without this, and a caption
# track whose boundaries cannot be trusted is not worth shipping.
#
# It costs ~4.25 GB in the image -- torch 1.70 GB, transformers + tokenizers
# 0.12 GB, and 2.43 GB of nvidia wheels torch pulls that CT2 does not need
# (nccl, cusparselt, cusolver, cusparse, cufft, nvrtc, curand, cupti). cuBLAS and
# cuDNN, the 1.84 GB CT2 already ships, are shared rather than added.
#
# ONNX Runtime was measured as the alternative and rejected: onnxruntime-gpu 1.29
# requires CUDA 13 (libcublasLt.so.13). Its CPU provider is numerically exact 
# (99.983% label agreement) and adds only ~30 MB, but runs 29.3s per file against torch's 0.5s on GPU.
PUNCTUATION = [
    'torch',
    'transformers',
]

# Reference implementation, kept for bench/ comparisons only.
# faster-whisper is 2.3x faster end to end and uses 2.3x less GPU (2556 vs 5774 MiB), 
# and it is the only backend with a VAD or a hookable fallback, so the clean-audio and reproducible profiles exist only there.
OPENAI_BACKEND = [
    'openai-whisper==20250625',
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
        *PUNCTUATION,
    ],
    extras_require={
        'ct2': CT2_BACKEND,
        'openai': OPENAI_BACKEND,
        'translate': TRANSLATE,
        'bench': BENCH,
        'all': CT2_BACKEND + OPENAI_BACKEND + TRANSLATE + BENCH,
    },
)
