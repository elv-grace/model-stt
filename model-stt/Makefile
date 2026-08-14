IMAGE_NAME := model-whisper-stt

.PHONY: test build weights bench

test:
	pytest tests

# populate the local weight cache (~/.cache/model-stt/whisper)
weights:
	python download_weights.py

# WEIGHTS=none|turbo-ct2|full-ct2|all, see build.sh
build:
	./build.sh

bench:
	python -m bench.run_bench --files test-files/*.m4a
