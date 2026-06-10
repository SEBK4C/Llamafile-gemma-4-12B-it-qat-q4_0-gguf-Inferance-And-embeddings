# Gemma 4 12B llamafile bridge — one server instance, two APIs.
#
# Pipeline:
#   make setup     initialize the llamafile submodule tree + cosmocc toolchain
#   make build     compile llamafile + zipalign from source (cosmopolitan APE)
#   make model     download gemma-4-12b-it-qat-q4_0 weights (+ vision mmproj)
#   make package   bake weights + args into a single-file gemma4-server.llamafile
#   make serve     run the dual-mode server (chat completions + embeddings)
#   make test      smoke-test both APIs against a running server

LLAMAFILE_DIR := vendor/llamafile
COSMO_MAKE    := $(LLAMAFILE_DIR)/.cosmocc/4.0.2/bin/make
JOBS          ?= $(shell sysctl -n hw.ncpu 2>/dev/null || nproc)

.PHONY: all setup build model package serve test clean

all: setup build model

setup:
	git submodule update --init vendor/llamafile
	cd $(LLAMAFILE_DIR) && git submodule update --init --depth 50 llama.cpp third_party/zipalign
	$(MAKE) -C $(LLAMAFILE_DIR) setup
	./scripts/apply-patches.sh

build:
	cd $(LLAMAFILE_DIR) && .cosmocc/4.0.2/bin/make -j$(JOBS) \
		o//llamafile/llamafile o//third_party/zipalign/zipalign
	mkdir -p bin
	cp $(LLAMAFILE_DIR)/o/llamafile/llamafile bin/llamafile
	cp $(LLAMAFILE_DIR)/o/third_party/zipalign/zipalign bin/zipalign

model:
	./scripts/fetch-model.sh

package:
	./scripts/package.sh

serve:
	./scripts/serve.sh

test:
	python3 tests/smoke_test.py

clean:
	rm -rf bin dist
