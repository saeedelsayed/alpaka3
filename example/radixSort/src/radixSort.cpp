#include "config.h"

#include <alpaka/alpaka.hpp>

#include <alpaka/onAcc/tag.hpp>

#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

#include <cstddef>
#include <cstdint>

constexpr uint32_t RADIX_BITS = 4;
constexpr uint32_t RADIX_SIZE = 1 << RADIX_BITS; // 16 bins
constexpr uint32_t BLOCK_SIZE = 16;


namespace alpaka::example::radixSort
{

        constexpr auto chunkSize = CVec<uint32_t, BLOCK_SIZE>{};

struct RadixCountKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::IDataSource auto const& data,
        alpaka::concepts::IMdSpan auto global_counts,
        uint32_t num_elements,
        uint32_t shift_bits) const
    {
        // Shared Memory (Private Histograms)
        auto& shared_hist = alpaka::onAcc::declareSharedVar<uint32_t[BLOCK_SIZE][RADIX_SIZE], __COUNTER__>(acc);

        // Initialize Shared Memory

        for(auto frameIdxMD :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::linearBlocksInGrid, IdxRange{acc[alpaka::frame::count].product()}))
        {
            for(auto frameElemIdxMD :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInBlock, IdxRange{acc[alpaka::frame::extent]}))
            {
                for(uint32_t bin = 0; bin < RADIX_SIZE; ++bin)
                {
                    shared_hist[frameElemIdxMD[0]][bin] = 0;
                }
            }

            // alpaka::onAcc::syncBlockThreads(acc);

            for(auto frameElemIdxMD :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::linearThreadsInBlock, IdxRange{acc[frame::extent].product()}))
            {
                auto const globalDataIdxMD = frameIdxMD * acc[frame::extent] + frameElemIdxMD;

                if(globalDataIdxMD.x() < num_elements)
                {
                    uint32_t const key = data[globalDataIdxMD];
                    uint32_t const digit = (key >> shift_bits) & 0x0F;

                    shared_hist[frameElemIdxMD[0]][digit]++;
                }
            }

            alpaka::onAcc::syncBlockThreads(acc);

            // Reduction (Summing columns)
            for(auto frameElemIdxMD :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::linearThreadsInBlock, IdxRange{acc[frame::extent].product()}))
            {
                if(frameElemIdxMD[0] < RADIX_SIZE)
                {
                    uint32_t block_total = 0;
                    for(uint32_t t = 0; t < BLOCK_SIZE; ++t)
                    {
                        block_total += shared_hist[t][frameElemIdxMD[0]];
                    }

                    // Write to Global Memory
                    uint32_t output_index = (frameIdxMD[0] * RADIX_SIZE) + frameElemIdxMD[0];
                    global_counts[output_index] = block_total;
                }
            }
        }
    }
};


void testRadixCount(alpaka::onHost::concepts::Device auto device, auto computeExec)
{
    // using namespace alpaka;

    using namespace alpaka;
    using namespace alpaka::onHost;

    std::vector<uint32_t> initial_data
        = {0, 0, 5, 5, 8, 7, 9, 8, 4, 5, 6, 2, 6, 8, 7, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 15, 14, 13, 13};

    uint32_t const num_elements = static_cast<uint32_t>(initial_data.size());

    // uint32_t const num_elements = (1024) / 4;

    std::cout << "Number of elements: " << num_elements << std::endl;
    auto h_data = onHost::allocHost<uint32_t>(Vec1D{num_elements});

    // std::mt19937 gen(42);
    // std::uniform_int_distribution<uint32_t> dist(0, 1000000);
    // for(uint32_t i = 0; i < num_elements; ++i) {
    //     h_data[i] = dist(gen);
    // }

    // Copy data into the Alpaka buffer
    std::cout << "Input Data: ";
    for(uint32_t i = 0; i < num_elements; ++i)
    {
        h_data[i] = initial_data[i];
        std::cout << h_data[i] << " ";
    }
    std::cout << "\n\n";

    auto extents = Vec<uint32_t, 1u>{num_elements};
    auto const numChunks = divCeil(extents, chunkSize);
    auto frameSpec = FrameSpec{numChunks, chunkSize};

    std::cout << frameSpec << std::endl;

    auto h_counts = onHost::allocHost<uint32_t>(Vec1D{numChunks.x() * RADIX_SIZE});

    onHost::Queue queue = device.makeQueue();

    auto d_data = onHost::allocLike(device, h_data);
    auto d_counts = onHost::allocLike(device, h_counts);

    onHost::memcpy(queue, d_data, h_data);
    onHost::memset(queue, d_counts, 0x00);

    onHost::wait(queue);
    auto const beginT = std::chrono::high_resolution_clock::now();

    queue.enqueue(computeExec, frameSpec, RadixCountKernel{}, d_data, d_counts, num_elements, 0u);

    onHost::wait(queue);
    auto const endT = std::chrono::high_resolution_clock::now();

 double kernelRuntime = std::chrono::duration<double, std::milli>(endT - beginT).count();

 std::cout << "Time taken by kernel: " << kernelRuntime << std::endl;

    onHost::memcpy(queue, h_counts, d_counts);
    onHost::wait(queue);

    // Aggregate GPU Results
    std::vector<uint32_t> gpu_total_histogram(RADIX_SIZE, 0);
    for(uint32_t blockID = 0; blockID < numChunks.x(); ++blockID)
    {
        for(uint32_t bin = 0; bin < RADIX_SIZE; ++bin)
        {
            uint32_t idx = blockID * RADIX_SIZE + bin;
            gpu_total_histogram[bin] += h_counts[idx];
        }
    }

    // Print GPU Table
    std::cout << "--------------------------------\n";
    std::cout << "GPU Count Table (Histogram):\n";
    std::cout << "--------------------------------\n";
    for(int i = 0; i < RADIX_SIZE; ++i)
    {
        std::cout << "Bin " << i << ": " << gpu_total_histogram[i] << "\n";
    }
    std::cout << "--------------------------------\n\n";

     std::vector<uint32_t> prefixScan(RADIX_SIZE, 0);
    for(uint32_t i = 0; i < RADIX_SIZE; i++)
    {
        if(i == 0)
        {
        prefixScan[i] = 0;
        }
        else
        { 
            for(int j = i-1; j >= 0; j--)
            {
                prefixScan[i]+= gpu_total_histogram[j];  
            }   
        }
    }

    // print PrefixScan Table
    std::cout << "--------------------------------\n";
    std::cout << "PrefixScan Table:\n";
    std::cout << "--------------------------------\n";
    for(uint32_t i = 0; i < RADIX_SIZE; ++i)
    {
        std::cout << "Bin " << i << ": " << prefixScan[i] << "\n";
    }
    std::cout << "--------------------------------\n";

    std::vector<uint32_t> ordered_data(num_elements, 0);

    for(uint32_t i = 0; i < num_elements; i++)
    {
        uint32_t digit = (initial_data[i] >> 0) & 0x0F;
        uint32_t index = prefixScan[digit];
        ordered_data[index] = initial_data[i];
        prefixScan[digit]++;
    }

    std::cout << "Ordered Data: ";
    for(uint32_t i = 0; i < num_elements; ++i)
    {
        std::cout << ordered_data[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "--------------------------------\n\n";

    std::cout << "Verifying results...\n";

    std::vector<uint32_t> cpu_histogram(RADIX_SIZE, 0);
    for(uint32_t i = 0; i < num_elements; ++i)
    {
        uint32_t digit = h_data[i] & 0x0F;
        cpu_histogram[digit]++;
    }

    bool correct = true;
    for(int i = 0; i < RADIX_SIZE; ++i)
    {
        if(cpu_histogram[i] != gpu_total_histogram[i])
        {
            std::cout << "Mismatch at bin " << i << ": CPU=" << cpu_histogram[i] << ", GPU=" << gpu_total_histogram[i]
                      << "\n";
            correct = false;
        }
    }

    if(correct)
    {
        std::cout << "SUCCESS! GPU counts match CPU.\n";
    }
    else
    {
        std::cout << "FAILURE! Results do not match.\n";
    }
}

int example(auto const deviceSpec, auto const computeExec)
{

    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);

    std::size_t n = devSelector.getDeviceCount();
    if(n == 0)
    {
        std::cerr << "No device found\n";
        return EXIT_FAILURE;
    }

    alpaka::onHost::Device device = devSelector.makeDevice(0);
    std::cout << "Device: " << alpaka::onHost::getName(device) << "\n\n";

    testRadixCount(device, computeExec);

    return EXIT_SUCCESS;
}

}
int main()
{
    using namespace alpaka;

    auto deviceSpec = onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
    auto executor = exec::gpuCuda;
    return alpaka::example::radixSort::example(deviceSpec, executor);
}

