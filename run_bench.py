import torch
from kernels.basic_matmul import matmul
from kernels.autotune_config import is_hip_mi200, is_cuda
import triton
from kernels.quantize import quantize
# import flashnn

def benchmark(M, N, K):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)

    rtol = 1e-2 if is_hip_mi200() else 0
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print(f"✅ M={M}, K={K}, N={N}, Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

    b = quantize(b, 8, dequantize=True)
    torch_output = torch.matmul(a, b)

    q_weight, zeros, scales = quantize(b, 8)

    # print(flashnn.GemmWeightOnly()(a, q_weight, scales, None, zeros))


    ms_torch = triton.testing.do_bench(lambda: torch.matmul(a, b))
    ms_triton = triton.testing.do_bench(lambda: matmul(a, b))
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    print("torch:", "%.2f"%perf(ms_torch), "TFLOPS")
    print("triton:", "%.2f"%perf(ms_triton), "TFLOPS")
    

for m in [4096,512,32,1]:
    for k,n in [(4096,4096),(4096,1024),(4096,14336),(14336,4096)]:
        benchmark(m,n,k)