SHELL := /bin/bash

BUILD_DIR ?= build
BUILD_TYPE ?= Release
CMAKE ?= cmake
CTEST ?= ctest
PYTHON ?= python3
JOBS ?= $(shell nproc)

CMAKE_ARGS ?= -DDFKV_BUILD_TESTS=ON -DDFKV_WITH_RDMA=ON -DDFKV_WITH_URING=ON
PY_TEST_ENV = PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(CURDIR)/integration/common/src DFKV_BUILD=$(CURDIR)/$(BUILD_DIR) DFKV_LIB=$(CURDIR)/$(BUILD_DIR)/libdfkv.so DFKV_STORE_ENGINE=file DFKV_SLAB_WRITE=buffered

.PHONY: all configure build test test-cpp test-python test-tools sanitize-test ci clean

all: build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_ARGS)

build: configure
	$(CMAKE) --build $(BUILD_DIR) -j$(JOBS)

test: test-cpp test-python test-tools

test-cpp: build
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure -j$(JOBS) \
		-E '^(python_plugin|python_client_metrics|python_telemetry|python_operational_tools|dfkvctl_all_groups|tool_smoke)$$'

test-python: build
	$(PY_TEST_ENV) $(PYTHON) -m unittest discover -s test/python -p 'test_*.py'
	$(PY_TEST_ENV) $(PYTHON) -m unittest discover -s integration/common/tests -p 'test_*.py'

test-tools: build
	bash -n test/tool_smoke.sh deploy/package_release.sh deploy/sync_telemetry.sh
	bash test/tool_smoke.sh $(BUILD_DIR)
	PYTHONPYCACHEPREFIX=$(CURDIR)/$(BUILD_DIR)/pycache $(PYTHON) -m py_compile deploy/*.py

sanitize-test:
	$(CMAKE) -S . -B $(BUILD_DIR)-sanitize -DCMAKE_BUILD_TYPE=Debug \
		-DDFKV_BUILD_TESTS=ON -DDFKV_WITH_RDMA=OFF -DDFKV_WITH_URING=OFF \
		-DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
		-DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
	$(CMAKE) --build $(BUILD_DIR)-sanitize -j$(JOBS)
	# ctypes cannot safely dlopen an ASan-instrumented libdfkv into stock Python.
	ASAN_OPTIONS=detect_leaks=1:allocator_may_return_null=1 UBSAN_OPTIONS=halt_on_error=1 \
		$(CTEST) --test-dir $(BUILD_DIR)-sanitize --output-on-failure -j$(JOBS) \
		-E '^python_plugin$$'

ci: build test

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR) $(BUILD_DIR)-sanitize
