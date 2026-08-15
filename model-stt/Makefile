IMAGE_NAME := model-whisper-stt

DUMMY := $(shell git submodule update --init 1>&2)
include buildscripts/Makefile.tagger-model

# `test` is the container end-to-end test, from buildscripts/Makefile.tagger-model.
# `build`/`container`/`deploy` come from buildscripts/Makefile.container, which
# already runs ./build.sh. Neither is redefined here: overriding an included
# target works but makes warns, and the recipes would be identical anyway.
.PHONY: pytest weights bench

# unit tests: no weights, no GPU
pytest:
	pytest tests

# populate the local weight cache (~/.cache/model-stt/whisper)
weights:
	python download_weights.py

bench:
	python -m bench.run_bench --files test-files/bench-files/*
