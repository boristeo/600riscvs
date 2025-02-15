#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel.hpp>
//#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>

#include <cstdio>

using namespace tt::tt_metal;
using namespace tt::tt_metal;

int main() {
  auto *device = CreateDevice(/*device_id=*/0);
  auto& hal = HalSingleton::getInstance();

  CommandQueue& cq = device->command_queue();
  Program program = CreateProgram();

  // 12x10?
  constexpr CoreCoord core_coord = {0, 1};


  // DataMovementConfig is interesting. Does every one of the little cpus get its own kernel? Or possible to load one to all?
  // I assume kernel goes somewhere in the 1mb core sram

  auto dm_kernel_conf = DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};

  auto kernel_src = KernelSource("loopback_dram_copy.cpp", KernelSource::FILE_PATH);
  puts("creating kernel");
  KernelHandle dram_copy_kernel_id = CreateKernel(program, kernel_src.source_, core_coord, dm_kernel_conf);

  //std::shared_ptr<Kernel> kernel = std::make_shared<DataMovementKernel>(kernel_src, CoreRangeSet{core_coord}, dm_kernel_conf);
  //auto dram_copy_kernel_id = 0;  // No kernels yet
  //uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
  //kernels_[index].insert({id, kernel});
  //kernel_groups_[index].resize(0);
  //core_to_kernel_group_index_table_[index].clear();


  //uint32_t programmable_core_index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
  //const KernelGroup* kernel_group = program.kernels_on_core(core_coord, programmable_core_index);

  // kernel group pointer should be null if no kernels loaded




  constexpr uint32_t single_tile_size = 2 * (32 * 32); // 2048
  constexpr uint32_t num_tiles = 50;
  constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles; // 100kb?
  // Guessing the minimum size is one tile?

  // Auto allocated? Any control over placement? What is page_size for?
  auto l1_config = InterleavedBufferConfig{.device=device, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::L1};
  auto l1_buffer = CreateBuffer(l1_config);

  auto dram_config = InterleavedBufferConfig{.device=device, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
  auto input_dram_buffer = CreateBuffer(dram_config);
  auto output_dram_buffer = CreateBuffer(dram_config);

  const uint32_t input_dram_buffer_addr = input_dram_buffer->address();
  const uint32_t output_dram_buffer_addr = output_dram_buffer->address();

  const uint32_t input_bank_id = 0;
  const uint32_t output_bank_id = 0;

  // Put some stuff in dram input area
  // What is 100 and why does it need time?
  std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

  puts("enq wbuf");
  // Non-blocking, does it start already?
  EnqueueWriteBuffer(cq, input_dram_buffer, input_vec, false);


  // Kernel args are just arbitrary vec of u32s
  const std::vector<uint32_t> runtime_args = {
    l1_buffer->address(),
    input_dram_buffer->address(),
    input_bank_id,
    output_dram_buffer->address(),
    output_bank_id,
    l1_buffer->size()
  };
  SetRuntimeArgs(program, dram_copy_kernel_id, core_coord, runtime_args);

  puts("enq prog");
  EnqueueProgram(cq, program, false);

  puts("finish");
  Finish(cq); // Nice

  std::vector<uint32_t> result_vec;
  puts("enq rbuf");
  EnqueueReadBuffer(cq,output_dram_buffer, result_vec, true);
  // true must mean blocking

  if (input_vec == result_vec)
    puts("PASS");
  else
    puts("MISMATCH");

  puts("close");
  // What if you don't close it?
  CloseDevice(device);

  return 0;
}
