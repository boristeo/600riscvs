loopback:
	clang++-17 -std=c++20 dram_loopback.cpp \
  -stdlib=libc++ \
  -lm -pthread -DFMT_HEADER_ONLY -fPIC \
  -fsized-deallocation \
  -Wno-c++11-narrowing \
  -L${TT_METAL_HOME}/build/lib -ltt_metal \
  -I${TT_METAL_HOME}/ \
  -I${TT_METAL_HOME}/tt_metal/api/ \
  -I${TT_METAL_HOME}/tt_metal/api/tt-metalium/ \
  -I${TT_METAL_HOME}/tt_metal/third_party/umd/device/api/ \
  -I${TT_METAL_HOME}/tt_metal/hostdevcommon/api/ \
  -I${TT_METAL_HOME}/tt_metal/third_party/tracy/public/ \
  -I${TT_METAL_HOME}/.cpmcache/fmt/73b5ec45edbd92babfd91c3777a9e1ab9cac8238/include/ \
  -I${TT_METAL_HOME}/.cpmcache/json/798e0374658476027d9723eeb67a262d0f3c8308/single_include/ \
  -I${TT_METAL_HOME}/.cpmcache/reflect/e75434c4c5f669e4a74e4d84e0a30d7249c1e66f/ \
  -I${TT_METAL_HOME}/.cpmcache/magic_enum/4d76fe0a5b27a0e62d6c15976d02b33c54207096/include/ \
  -I${TT_METAL_HOME}/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/core/include/
