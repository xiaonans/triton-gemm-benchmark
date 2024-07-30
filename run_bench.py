import torch
from kernels.basic_matmul import matmul
from kernels.autotune_config import is_hip_mi200, is_cuda
import triton
from kernels.quantize import quantize
import flashnn

def benchmark(M, N, K):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)

    rtol = 1e-2 if is_hip_mi200() else 0
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

    b = quantize(b, 8, dequantize=True)
    print(torch_output)
    torch_output = torch.matmul(a, b)
    print(torch_output)

    q_weight, zeros, scales = quantize(b, 8)

    print(flashnn.GemmWeightOnly()(a, q_weight, scales, None, zeros))


    ms_torch = triton.testing.do_bench(lambda: torch.matmul(a, b))
    ms_triton = triton.testing.do_bench(lambda: matmul(a, b))
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    print("torch:", perf(ms_torch), "TFLOPS")
    print("triton:", perf(ms_triton), "TFLOPS")
    


benchmark(1,8,256)