#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>

#include <cstdio>

using namespace tt::tt_metal;

int main() {
  Device *device = CreateDevice(/*device_id=*/0);
  CommandQueue& cq = detail::GetCommandQueue(device);
  Program program = CreateProgram();

  // 12x10?
  constexpr CoreCoord coord = {0, 0};

  // DataMovementConfig is interesting. Does every one of the little cpus get its own kernel? Or possible to load one to all?
  // I assume kernel goes somewhere in the 1mb core sram
  KernelHandle dram_copy_kernel_id = CreateKernel(program, "loopback_dram_copy.cpp", coord,
      DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});


  constexpr uint32_t single_tile_size = 2 * (32 * 32); // 2048
  constexpr uint32_t num_tiles = 50;
  constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles; // 100kb?
  // Guessing the minimum size is one tile?

  // Auto allocated? Any control over placement? What is page_size for?
  auto l1_config = tt_metal::InterleavedBufferConfig{.device=device, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = tt_metal::BufferType::L1};
  Buffer l1_buffer = CreateBuffer(l1_config);

  auto dram_config = tt_metal::InterleavedBufferConfig{.device=device, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = tt_metal::BufferType::DRAM};
  Buffer input_dram_buffer = CreateBuffer(dram_config);
  Buffer output_dram_buffer = CreateBuffer(dram_config);

  const uint32_t input_dram_buffer_addr = input_dram_buffer.address();
  const uint32_t output_dram_buffer_addr = output_dram_buffer.address();

  const uint32_t input_bank_id = 0;
  const uint32_t output_bank_id = 0;

  // Put some stuff in dram input area
  // What is 100 and why does it need time?
  std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
  // Non-blocking, does it start already?
  EnqueueWriteBuffer(cq, input_dram_buffer, input_vec, false);


  // Kernel args are just arbitrary vec of u32s
  const std::vector<uint32_t> runtime_args = {
    l1_buffer.address(),
    input_dram_buffer.address(),
    input_bank_id,
    output_dram_buffer.address(),
    output_bank_id,
    l1_buffer.size()
  };
  SetRuntimeArgs(program, dram_copy_kernel_id, core, runtime_args);

  EnqueueProgram(cq, program, false);
  Finish(cq); // Nice

  std::vector<uint32_t> result_vec;
  EnqueueReadBuffer(cq,output_dram_buffer, result_vec, true);
  // true must mean blocking

  if (input_vec == result_vec)
    puts("PASS");
  else
    puts("MISMATCH");

  // What if you don't close it?
  CloseDevice(device);

  return 0;
}
