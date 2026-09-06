# dfkv — standalone distributed KV cache for SGLang HiCache (no brpc/MDS/S3 deps).
cmake_minimum_required(VERSION 3.20)
# Single source of truth for the version is the VERSION file (also shipped in
# the release tarball). CMake reads it here so `project(dfkv VERSION ...)`
# stays in sync with VERSION without a bump-both-files step (which bit v1.12.0:
# the VERSION file was bumped but project() wasn't, so the binary's --version
# reported 1.11.0). Strip trailing whitespace/newline.
file(READ "${CMAKE_SOURCE_DIR}/VERSION" _dfkv_ver_raw)
string(STRIP "${_dfkv_ver_raw}" _dfkv_ver)
project(dfkv VERSION "${_dfkv_ver}" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

option(DFKV_BUILD_TESTS "Build dfkv tests" ON)
option(DFKV_STATIC_LIBSTDCXX "Statically link libstdc++/libgcc for portable artifacts" ON)
option(DFKV_WITH_RDMA "Build the RDMA transport (needs libibverbs)" OFF)
# Additive, env-gated io_uring async-GET disk-read path in the RDMA server. When
# ON, links liburing and compiles the DFKV_WITH_URING code; the path is still
# inert unless the server is started with DFKV_SERVER_URING=1. When OFF (default)
# the io_uring code is #ifdef'd out and the env flag is a no-op. Requires RDMA.
option(DFKV_WITH_URING "Build the io_uring async GET path in the RDMA server (needs liburing)" OFF)
if(DFKV_STATIC_LIBSTDCXX)
  add_link_options(-static-libstdc++ -static-libgcc)
endif()
if(DFKV_WITH_URING AND NOT DFKV_WITH_RDMA)
  message(FATAL_ERROR "DFKV_WITH_URING requires DFKV_WITH_RDMA=ON (the io_uring path lives in the RDMA server)")
endif()
if(DFKV_WITH_URING)
  # Prefer the STATIC archive: the portable release must not add a liburing.so
  # runtime dependency on deploy nodes (liburing is a thin syscall wrapper, so
  # static linking costs ~50 KB and carries no glibc-floor risk).
  find_library(URING_LIB NAMES liburing.a uring)
  if(NOT URING_LIB)
    message(FATAL_ERROR "DFKV_WITH_URING=ON but liburing not found (install liburing-dev)")
  endif()
  message(STATUS "liburing: ${URING_LIB}")
endif()

# --- portable core library (everything in src/ except *_main.cc + gated rdma) ---
file(GLOB_RECURSE DFKV_SRCS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
list(FILTER DFKV_SRCS EXCLUDE REGEX "_main\\.cc$")
if(NOT DFKV_WITH_RDMA)
  list(REMOVE_ITEM DFKV_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/transport/rdma_verbs.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/transport/rdma_topology.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/transport/rdma_transport.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cache/rdma_server.cc)
endif()

add_library(dfkv_core STATIC ${DFKV_SRCS})
target_include_directories(dfkv_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(dfkv_core PRIVATE _GNU_SOURCE  # O_DIRECT/fallocate in src/cache/kv_store.cc
  DFKV_VERSION="${PROJECT_VERSION}")
find_package(Threads REQUIRED)
# src/client/cuda_ipc.cc dlopen()s libcuda.so.1 (needs libdl); node_dedup*.cc use
# POSIX shm_open/shm_unlink (needs librt on glibc < 2.34). PUBLIC so both propagate
# to every executable linking the dfkv_core static archive.
target_link_libraries(dfkv_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS} rt)
if(DFKV_WITH_RDMA)
  target_compile_definitions(dfkv_core PUBLIC DFKV_WITH_RDMA)
  target_link_libraries(dfkv_core PUBLIC ibverbs)
endif()
if(DFKV_WITH_URING)
  target_compile_definitions(dfkv_core PUBLIC DFKV_WITH_URING)
  target_link_libraries(dfkv_core PUBLIC ${URING_LIB})
endif()

# C ABI shared lib for the Python (ctypes) plugin
add_library(dfkv SHARED ${DFKV_SRCS})
target_include_directories(dfkv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(dfkv PRIVATE _GNU_SOURCE  # O_DIRECT/fallocate in src/cache/kv_store.cc
  DFKV_VERSION="${PROJECT_VERSION}")
target_link_libraries(dfkv PUBLIC Threads::Threads ${CMAKE_DL_LIBS} rt)  # cuda_ipc dlopen->libdl, node_dedup shm_*->librt
set_target_properties(dfkv PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION 2)
if(DFKV_WITH_RDMA)
  target_compile_definitions(dfkv PUBLIC DFKV_WITH_RDMA)
  target_link_libraries(dfkv PUBLIC ibverbs)
endif()
if(DFKV_WITH_URING)
  target_compile_definitions(dfkv PUBLIC DFKV_WITH_URING)
  target_link_libraries(dfkv PUBLIC ${URING_LIB})
endif()

# daemons (entry points live with their subsystems) + CLI tools
add_executable(dfkv_server ${CMAKE_CURRENT_SOURCE_DIR}/src/cache/dfkv_server_main.cc)
target_link_libraries(dfkv_server PRIVATE dfkv_core)
add_executable(dfkv_mds ${CMAKE_CURRENT_SOURCE_DIR}/src/mds/dfkv_mds_main.cc)
target_link_libraries(dfkv_mds PRIVATE dfkv_core)
add_executable(dfkv_smoke ${CMAKE_CURRENT_SOURCE_DIR}/src/tools/dfkv_smoke_main.cc)
target_link_libraries(dfkv_smoke PRIVATE dfkv_core)
add_executable(dfkvctl ${CMAKE_CURRENT_SOURCE_DIR}/src/tools/dfkvctl_main.cc)
target_link_libraries(dfkvctl PRIVATE dfkv_core)
add_executable(dfkv_bench ${CMAKE_CURRENT_SOURCE_DIR}/src/tools/dfkv_bench_main.cc)
target_link_libraries(dfkv_bench PRIVATE dfkv_core)
add_executable(dfkv_discover_smoke ${CMAKE_CURRENT_SOURCE_DIR}/src/tools/dfkv_discover_smoke_main.cc)
target_link_libraries(dfkv_discover_smoke PRIVATE dfkv_core)
if(DFKV_WITH_RDMA)
  # In-process server+fan-out client memory benchmark; needs real verbs.
  add_executable(dfkvleasebench
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tools/dfkvleasebench_main.cc)
  target_link_libraries(dfkvleasebench PRIVATE dfkv_core)
  install(TARGETS dfkvleasebench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# Real-HW lockstep repro for the GPU rendezvous (needs CUDA devices at
# runtime; builds everywhere — CUDA is dlopen'd).
add_executable(dfkv_gpu_dedup_repro ${CMAKE_CURRENT_SOURCE_DIR}/src/tools/dfkv_gpu_dedup_repro_main.cc)
target_link_libraries(dfkv_gpu_dedup_repro PRIVATE dfkv_core)

include(GNUInstallDirs)
install(TARGETS dfkv dfkv_server dfkv_mds dfkv_smoke dfkvctl dfkv_bench
        dfkv_discover_smoke dfkv_gpu_dedup_repro dfkv_core
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY src/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dfkv FILES_MATCHING PATTERN "*.h")
# The standalone HiCache plugin imports these sibling modules at import time.
install(FILES
  integration/hicache/dfkv_hicache.py
  integration/hicache/dfkv_access_log.py
  integration/hicache/dfkv_hot_config.py
  integration/hicache/dfkv_metrics.py
  DESTINATION ${CMAKE_INSTALL_DATADIR}/dfkv/python)
# Canonical binary namespace/client ABI package shared by every connector.
install(DIRECTORY integration/common/src/dfkv_common
  DESTINATION ${CMAKE_INSTALL_DATADIR}/dfkv/python
  FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)
# Shared telemetry package (push metrics and tracing).
install(DIRECTORY integration/hicache/dfkv_telemetry
  DESTINATION ${CMAKE_INSTALL_DATADIR}/dfkv/python
  FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)
# Ship the supported operational scripts, deployment templates, connector
# sources, documentation, and referenced C++/test sources in one inspectable
# repository layout. Relative links under docs/ therefore remain valid offline.
install(DIRECTORY src docs deploy integration test .github
  DESTINATION ${CMAKE_INSTALL_DATADIR}/dfkv
  USE_SOURCE_PERMISSIONS
  PATTERN "__pycache__" EXCLUDE
  PATTERN ".pytest_cache" EXCLUDE
  PATTERN ".mypy_cache" EXCLUDE
  PATTERN "*.egg-info" EXCLUDE
  PATTERN "build" EXCLUDE
  PATTERN "dist" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)
install(FILES CMakeLists.txt Makefile Dockerfile .dockerignore
              README.md CHANGELOG.md VERSION LICENSE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/dfkv)

# --- tests ---
if(DFKV_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)

  file(GLOB_RECURSE DFKV_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/*_test.cc)
  # rdma_loopback_test needs the RDMA transport/server symbols -> RDMA build only.
  list(REMOVE_ITEM DFKV_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/test/transport/rdma_loopback_test.cc)
  list(REMOVE_ITEM DFKV_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/test/transport/rdma_lease_loopback_test.cc)
  list(REMOVE_ITEM DFKV_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/test/transport/qp_depth_negotiation_test.cc)
  list(REMOVE_ITEM DFKV_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/test/transport/rdma_topology_test.cc)
  # Backend semantics have dedicated tests below. Keep unrelated functional
  # tests on the explicit file backend so a full CTest run does not accumulate
  # 1 GiB production extents in /tmp and become order/host-capacity dependent.
  set(DFKV_UNIT_TEST_ENV
      "DFKV_STORE_ENGINE=file;DFKV_SLAB_WRITE=buffered")
  foreach(t ${DFKV_TESTS})
    get_filename_component(name ${t} NAME_WE)
    add_executable(${name} ${t})
    target_link_libraries(${name} PRIVATE dfkv_core GTest::gtest_main)
    # node_dedup_gpu_test drives real CUDA (arenas, contexts, cross-process
    # IPC via fork+exec). Under `ctest -j N` on the B200 gate it otherwise runs
    # concurrently with the RDMA datapath tests, and the child of the IPC test
    # fails opening the peer handle under that GPU contention (flaked the gate
    # even with GPUs otherwise free). A shared RESOURCE_LOCK serializes every
    # GPU/RDMA-touching test against each other while the rest still parallelize.
    if(name STREQUAL "node_dedup_gpu_test")
      gtest_discover_tests(${name} PROPERTIES
        RESOURCE_LOCK gpu_rdma_hw
        ENVIRONMENT "${DFKV_UNIT_TEST_ENV}")
    else()
      gtest_discover_tests(${name} PROPERTIES
        ENVIRONMENT "${DFKV_UNIT_TEST_ENV}")
    endif()
  endforeach()
  if(DFKV_WITH_RDMA)  # exercised in CI under Soft-RoCE (rdma_rxe); skips if no device
    add_executable(rdma_loopback_test ${CMAKE_CURRENT_SOURCE_DIR}/test/transport/rdma_loopback_test.cc)
    add_executable(rdma_lease_loopback_test
      ${CMAKE_CURRENT_SOURCE_DIR}/test/transport/rdma_lease_loopback_test.cc)
    target_link_libraries(rdma_loopback_test PRIVATE dfkv_core GTest::gtest_main)
    target_link_libraries(rdma_lease_loopback_test PRIVATE dfkv_core GTest::gtest_main)
    gtest_discover_tests(rdma_loopback_test PROPERTIES
      RESOURCE_LOCK gpu_rdma_hw
      ENVIRONMENT "${DFKV_UNIT_TEST_ENV}")
    gtest_discover_tests(rdma_lease_loopback_test PROPERTIES
      RESOURCE_LOCK gpu_rdma_hw
      ENVIRONMENT "${DFKV_UNIT_TEST_ENV}")
    add_executable(qp_depth_negotiation_test ${CMAKE_CURRENT_SOURCE_DIR}/test/transport/qp_depth_negotiation_test.cc)
    target_link_libraries(qp_depth_negotiation_test PRIVATE dfkv_core GTest::gtest_main)
    gtest_discover_tests(qp_depth_negotiation_test PROPERTIES
      ENVIRONMENT "${DFKV_UNIT_TEST_ENV}")
    add_executable(rdma_topology_test ${CMAKE_CURRENT_SOURCE_DIR}/test/transport/rdma_topology_test.cc)
    target_link_libraries(rdma_topology_test PRIVATE dfkv_core GTest::gtest_main)
    gtest_discover_tests(rdma_topology_test PROPERTIES
      ENVIRONMENT "${DFKV_UNIT_TEST_ENV}")
  endif()

  # Python plugin test (ctypes -> libdfkv.so -> dfkv_server). No GPU/torch.
  find_program(PYTHON3 python3)
  if(PYTHON3)
    add_test(NAME python_plugin
      COMMAND ${PYTHON3} -m unittest test_dfkv_identity test_dfkv_hicache
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test/python)
    set_tests_properties(python_plugin PROPERTIES
      ENVIRONMENT "DFKV_BUILD=${CMAKE_BINARY_DIR};DFKV_LIB=${CMAKE_BINARY_DIR}/libdfkv.so;DFKV_STORE_ENGINE=file;DFKV_SLAB_WRITE=buffered;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/integration/common/src")
    add_test(NAME python_client_metrics
      COMMAND ${PYTHON3} -m unittest discover
              -s ${CMAKE_CURRENT_SOURCE_DIR}/integration/common/tests
              -p test_client_metrics.py)
    set_tests_properties(python_client_metrics PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/integration/common/src")
    add_test(NAME python_common_rail_affinity
      COMMAND ${PYTHON3} -m unittest discover
              -s ${CMAKE_CURRENT_SOURCE_DIR}/integration/common/tests
              -p test_rail_affinity.py)
    set_tests_properties(python_common_rail_affinity PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/integration/common/src")
    add_test(NAME python_lmcache_rail_affinity
      COMMAND ${PYTHON3} -m unittest discover
              -s ${CMAKE_CURRENT_SOURCE_DIR}/integration/lmcache/tests
              -p test_rail_affinity.py)
    set_tests_properties(python_lmcache_rail_affinity PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/integration/common/src")
    add_test(NAME python_vllm_rail_affinity
      COMMAND ${PYTHON3} -m unittest discover
              -s ${CMAKE_CURRENT_SOURCE_DIR}/integration/vllm/tests
              -p test_rail_affinity.py)
    set_tests_properties(python_vllm_rail_affinity PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/integration/common/src:${CMAKE_CURRENT_SOURCE_DIR}/integration/vllm/src")
    add_test(NAME python_vllm_client_ranks
      COMMAND ${PYTHON3} -m unittest discover
              -s ${CMAKE_CURRENT_SOURCE_DIR}/integration/vllm/tests
              -p test_client_ranks.py)
    set_tests_properties(python_vllm_client_ranks PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/integration/common/src:${CMAKE_CURRENT_SOURCE_DIR}/integration/vllm/src")
    # Shared telemetry layer: pure-python, no build/lib needed (OTel-push test
    # self-skips when the SDK is absent). Includes the vendored-copy drift guard.
    add_test(NAME python_telemetry
      COMMAND ${PYTHON3} -m unittest test_dfkv_telemetry test_telemetry_vendor_sync
              test_observability_contract test_release_contract
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test/python)
    set_tests_properties(python_telemetry PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/integration/common/src")
    add_test(NAME python_operational_tools
      COMMAND ${PYTHON3} -m unittest test_dfkv_operational_tools
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test/python)
    add_test(NAME dfkvctl_all_groups
      COMMAND ${PYTHON3}
              ${CMAKE_CURRENT_SOURCE_DIR}/test/python/test_dfkvctl_all_groups.py
              ${CMAKE_BINARY_DIR})
  endif()

  # CLI tools integration smoke (dfkv_server + dfkv_smoke + dfkvctl)
  add_test(NAME tool_smoke
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test/tool_smoke.sh ${CMAKE_BINARY_DIR})
endif()

# --- packaging (deb/rpm via CPack) ---
set(CPACK_PACKAGE_NAME "dfkv")
set(CPACK_PACKAGE_VENDOR "ketor")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Distributed KV cache for SGLang HiCache")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_CONTACT "d.ketor@gmail.com")
set(CPACK_GENERATOR "TGZ")  # default; deb/rpm chosen explicitly via `cpack -G DEB|RPM`
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
include(CPack)
