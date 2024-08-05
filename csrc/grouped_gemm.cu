#include "cutlass/layout/matrix.h"
#include "grouped_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <cstddef>
#include <torch/extension.h>
#include <type_traits>

#include "cutlass/bfloat16.h"
#include "cutlass/complex.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
namespace grouped_gemm {

#define NUM_STREAM 4

#define CUDA_CALL(code)                                                        \
  do {                                                                         \
    cudaError_t status = code;                                                 \
    std::string err = cudaGetErrorString(status);                              \
    TORCH_CHECK(status == cudaSuccess, err);                                   \
  } while (0)

template <bool TransA = false, bool TransB = false>
class CutlassGroupedGemmMoE {
  static_assert(not(TransA and TransB),
                "Not implemented when TransA=True and TransB=True");
  using LayoutA = std::conditional_t<TransA, ::cutlass::layout::ColumnMajor,
                                     ::cutlass::layout::RowMajor>;
  using LayoutB = std::conditional_t<TransB, ::cutlass::layout::ColumnMajor,
                                     ::cutlass::layout::RowMajor>;

  // TODO(tgale): Update this for SM90 when it's supported by CUTLASS.
  using GroupedGemmKernelNN =
      typename cutlass::gemm::kernel::DefaultGemmGrouped<
          // Non-transposed A operand.
          ::cutlass::bfloat16_t, LayoutA, ::cutlass::ComplexTransform::kNone, 8,
          // Non-transposed B operand.
          ::cutlass::bfloat16_t, LayoutB, ::cutlass::ComplexTransform::kNone, 8,
          // C operand.
          ::cutlass::bfloat16_t, ::cutlass::layout::RowMajor, float,
          ::cutlass::arch::OpClassTensorOp, ::cutlass::arch::Sm80,
          ::cutlass::gemm::GemmShape<128, 128, 32>,
          ::cutlass::gemm::GemmShape<64, 64, 32>,
          ::cutlass::gemm::GemmShape<16, 8, 16>,
          ::cutlass::epilogue::thread::LinearCombination<::cutlass::bfloat16_t,
                                                         8, float, float>,
          // NOTE: Threadblock swizzling is currently not supported by CUTLASS's
          // grouped kernels. This parameter is passed in at present to match
          // the APIs of other kernels. The parameter is unused within the
          // kernel.
          ::cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
          // TODO(tgale): Experiment with GroupScheduleMode.
          // TODO(tgale): Tune this for SM90.
          4>::GemmKernel;
  using GemmGroupedNN =
      ::cutlass::gemm::device::GemmGrouped<GroupedGemmKernelNN>;

  template <typename T>
  torch::Tensor CopyToDevice(const std::vector<T> &x,
                             const torch::Device &device) {
    size_t bytes = x.size() * sizeof(T);
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
    torch::Tensor out = torch::empty(bytes, options);

    CUDA_CALL(cudaMemcpyAsync(out.data_ptr(), x.data(), bytes,
                              cudaMemcpyHostToDevice,
                              c10::cuda::getCurrentCUDAStream()));
    return out;
  }
  std::vector<cutlass::gemm::GemmCoord>
  MakeProblemSizes(torch::Tensor a, torch::Tensor b, torch::Tensor batch_sizes,
                   bool skip_zero = false) {
    const size_t num_experts = batch_sizes.size(0);
    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    auto bs_ptr = batch_sizes.data_ptr<int64_t>();
    for (int i = 0; i < num_experts; ++i) {
      int m = 0, n = 0, k = 0;
      if constexpr (TransA) {
        // A(49152*2048) B(48152*2816) C(64*2048*2816)
        m = a.size(1), n = b.size(1), k = bs_ptr[i];
      } else if constexpr (TransB) {
        // A(49152*2816) B(64*2048*2816) C(49152*2048)
        k = a.size(1), n = b.size(1), m = bs_ptr[i];
      } else {
        // A(49152*2048) B(64*2048*2816) C(49152*2816)
        k = b.size(1), n = b.size(2), m = bs_ptr[i];
      }
      if (skip_zero and bs_ptr[i] == 0) {
        continue;
      }
      problem_sizes.push_back({m, n, k});
    }
    return problem_sizes;
  }

  template <typename Gemm>
  typename Gemm::Arguments MakeArguments(torch::Tensor a, torch::Tensor b,
                                         torch::Tensor c,
                                         torch::Tensor batch_sizes) {
    auto problem_sizes_host_fake = MakeProblemSizes(a, b, batch_sizes);
    auto problem_sizes_host = MakeProblemSizes(a, b, batch_sizes, true);

    // Calculate the number of threadblocks to use and validate the result.
    int64_t num_experts = problem_sizes_host.size();

    // NOTE: This is borrowed from FasterTransformer.
    int threadblock_count =
        Gemm::sufficient(problem_sizes_host.data(), num_experts);
    if (!threadblock_count) {
      TORCH_CHECK(false, "Grouped GEMM execution not possible with HW");
    }

    // Create the host arrays of leading dimension data and pointer data.
    using LayoutA = typename Gemm::LayoutA;
    using LayoutB = typename Gemm::LayoutB;
    using LayoutC = typename Gemm::LayoutC;

    std::vector<int64_t> lda_host(num_experts), offsets_a(num_experts);
    std::vector<int64_t> ldb_host(num_experts), offsets_b(num_experts);
    std::vector<int64_t> ldc_host(num_experts), offsets_c(num_experts);
    int64_t elements_a = 0, elements_b = 0, elements_c = 0;

    using ElementA = typename Gemm::ElementA;
    using ElementB = typename Gemm::ElementB;
    using ElementC = typename Gemm::ElementC;
    std::vector<ElementA *> ptr_a_host(num_experts);
    std::vector<ElementB *> ptr_b_host(num_experts);
    std::vector<ElementC *> ptr_c_host(num_experts);

    for (int i = 0, j = 0; j < problem_sizes_host_fake.size(); ++j) {
      auto problem = problem_sizes_host_fake[j];
      if (problem.m() != 0 and problem.n() != 0 and problem.k() != 0) {
        lda_host[i] = LayoutA::packed({problem.m(), problem.k()}).stride(0);
        ldb_host[i] = LayoutB::packed({problem.k(), problem.n()}).stride(0);
        ldc_host[i] = LayoutC::packed({problem.m(), problem.n()}).stride(0);

        offsets_a[i] = elements_a;
        offsets_b[i] = elements_b;
        offsets_c[i] = elements_c;

        ptr_a_host[i] = (ElementA *)a.data_ptr() + offsets_a[i];
        ptr_b_host[i] = (ElementB *)b.data_ptr() + offsets_b[i];
        ptr_c_host[i] = (ElementC *)c.data_ptr() + offsets_c[i];
        i++;
      }
      elements_a += problem.m() * problem.k();
      elements_b += problem.k() * problem.n();
      elements_c += problem.m() * problem.n();
    }

    // Copy the problem sizes, pointers and leading dimension data to the
    // device.
    torch::Tensor lda = CopyToDevice(lda_host, a.device());
    torch::Tensor ldb = CopyToDevice(ldb_host, a.device());
    torch::Tensor ldc = CopyToDevice(ldc_host, a.device());
    torch::Tensor ptr_a = CopyToDevice(ptr_a_host, a.device());
    torch::Tensor ptr_b = CopyToDevice(ptr_b_host, a.device());
    torch::Tensor ptr_c = CopyToDevice(ptr_c_host, a.device());
    torch::Tensor problem_sizes = CopyToDevice(problem_sizes_host, a.device());

    typename Gemm::EpilogueOutputOp::Params epilogue_op(/*alpha=*/1.0f,
                                                        /*beta=*/0.0f);
    typename Gemm::Arguments arguments(
        (cutlass::gemm::GemmCoord *)problem_sizes.data_ptr(), (int)num_experts,
        (int)threadblock_count, epilogue_op, (ElementA **)ptr_a.data_ptr(),
        (ElementB **)ptr_b.data_ptr(), (ElementC **)ptr_c.data_ptr(),
        (ElementC **)ptr_c.data_ptr(),
        /*lda=*/(int64_t *)lda.data_ptr(),
        /*ldb=*/(int64_t *)ldb.data_ptr(),
        /*ldc=*/(int64_t *)ldc.data_ptr(),
        /*ldd=*/(int64_t *)ldc.data_ptr(),
        (cutlass::gemm::GemmCoord *)problem_sizes_host.data());
    return arguments;
  }

public:
  static CutlassGroupedGemmMoE<TransA, TransB> &Ins() {
    static CutlassGroupedGemmMoE<TransA, TransB> ins_;
    return ins_;
  }
  void Calculate(torch::Tensor a, torch::Tensor b, torch::Tensor c,
                 torch::Tensor batch_sizes) {
    using Gemm = GemmGroupedNN;
    Gemm gemm;

    auto arguments = MakeArguments<Gemm>(a, b, c, batch_sizes);
    int64_t workspace_size = gemm.get_workspace_size(arguments);
    auto options =
        torch::TensorOptions().dtype(torch::kInt8).device(a.device());
    torch::Tensor workspace = torch::empty(workspace_size, options);

    // Initialize the kernel.
    if (gemm.initialize(arguments, workspace.data_ptr()) !=
        cutlass::Status::kSuccess) {
      TORCH_CHECK(false, "Failed to initialize CUTLASS Grouped GEMM");
    }

    // Execute the kernel in the current stream.
    if (gemm.run(c10::cuda::getCurrentCUDAStream()) !=
        cutlass::Status::kSuccess) {
      TORCH_CHECK(false, "Failed to run CUTLASS Grouped GEMM");
    }
    return;
  }
};

// NOTE: We only support dynamic group sizes for the 'a' tensor. Tensor 'b' is
// assumed to be batched with fixed sized batches.
//
// TODO(tgale): Validate alignment is true for every batch element.
void GroupedGemm(torch::Tensor a, torch::Tensor b, torch::Tensor c,
                 torch::Tensor batch_sizes, bool trans_a, bool trans_b) {
  // NOTE: We only support 'trans_a' or 'trans_b', not both.
  TORCH_CHECK(!(trans_a && trans_b));

  // We expect the batch_sizes on CPU.
  TORCH_CHECK(batch_sizes.is_cpu());
  TORCH_CHECK(batch_sizes.ndimension() == 1);
  TORCH_CHECK(batch_sizes.scalar_type() == torch::kInt64);

  // We expected a CUDA tensor with two dimensions and shape
  // (tokens, hidden_in) for 'a'.
  TORCH_CHECK(a.is_cuda());
  TORCH_CHECK(a.ndimension() == 2);
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16);

  // Defer to the variable 'k' helper for the rest of the op.
  if (trans_a) {
    CutlassGroupedGemmMoE<true, false>::Ins().Calculate(a, b, c, batch_sizes);
    return;
  }

  // We expected a CUDA tensor with three dimensions and shape
  // (num_experts, hidden_in, hidden_out) for 'b'.
  TORCH_CHECK(b.is_cuda());
  TORCH_CHECK(b.ndimension() == 3);
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16);

  // Validate the contraction dimensions match.
  int64_t tokens = a.size(0), num_experts = b.size(0);
  int64_t hidden_in = trans_b ? b.size(2) : b.size(1);
  int64_t hidden_out = trans_b ? b.size(1) : b.size(2);
  TORCH_CHECK(hidden_in == a.size(1));

  // Validate that we have one size per expert.
  TORCH_CHECK(batch_sizes.size(0) == num_experts);

  // Validate the output shape.
  TORCH_CHECK(c.is_cuda());
  TORCH_CHECK(c.ndimension() == 2);
  TORCH_CHECK(c.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(c.size(0) == tokens);
  TORCH_CHECK(c.size(1) == hidden_out);

  // NOTE: We support transposition through the 'trans_b' flag.
  TORCH_CHECK(a.is_contiguous());
  TORCH_CHECK(b.is_contiguous());

  if (trans_b) {
    CutlassGroupedGemmMoE<false, true>::Ins().Calculate(a, b, c, batch_sizes);
    return;
  } else {
    CutlassGroupedGemmMoE<false, false>::Ins().Calculate(a, b, c, batch_sizes);
  }
  return;
}

} // namespace grouped_gemm_cutlass2
