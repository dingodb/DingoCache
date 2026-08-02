# Portable dfkv build with RDMA, pinned to an OLD glibc for wide compatibility.
#
# Why ubuntu:22.04 (glibc 2.35): a dynamically-linked binary's glibc floor = the
# glibc it was BUILT against. Building on a newer host (e.g. ubuntu:24.04 / glibc
# 2.39) makes artifacts that FAIL on older nodes with "GLIBC_2.3x not found".
# 2.35 runs on RHEL9/Ubuntu22.04+ and most current GPU nodes. Build once here, deploy
# on any node with glibc >= 2.35 -- no per-node rebuild.
#
# DFKV_STATIC_LIBSTDCXX folds libstdc++/libgcc into the artifacts (those are NOT
# glibc; static-linking them removes a separate runtime dep). libibverbs CANNOT
# be static (it dlopen()s provider drivers at runtime), so the run node still
# needs rdma-core / libibverbs installed.
ARG DFKV_BASE_IMAGE=m.daocloud.io/docker.io/library/ubuntu:22.04
FROM ${DFKV_BASE_IMAGE} AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build g++ git ca-certificates libibverbs-dev liburing-dev && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DDFKV_BUILD_TESTS=OFF \
    -DDFKV_WITH_RDMA=ON -DDFKV_WITH_URING=ON -DDFKV_STATIC_LIBSTDCXX=ON && \
    cmake --build build -j && cmake --install build --prefix /out && \
    (strip /out/bin/* /out/lib/*.so* 2>/dev/null || true)
# Artifacts (portable, glibc>=2.35): /out/bin/{dfkv_server,dfkv_mds,dfkv_bench,
# dfkv_smoke,dfkvctl} and /out/lib/libdfkv.so. Extract with:
#   docker build -t dfkv-build --target build . && \
#   id=$(docker create dfkv-build) && docker cp $id:/out ./dist && docker rm $id

FROM ${DFKV_BASE_IMAGE} AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    rdma-core libibverbs1 && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /usr/local/
RUN ldconfig
EXPOSE 12000
ENTRYPOINT ["dfkv_server"]
CMD ["--dir", "/data", "--port", "12000", "--cap", "6442450944"]
