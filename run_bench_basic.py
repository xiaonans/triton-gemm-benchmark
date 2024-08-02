import torch
from kernels.basic_matmul import matmul
from kernels.autotune_config import is_hip_mi200, is_cuda
import triton
from kernels.quantize import quantize
# import flashnn

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def benchmark(M, N, K):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)

    rtol = 1e-2 #if is_hip_mi200() else 0
    try:
        torch.testing.assert_close(torch_output, triton_output, rtol=rtol, atol=1e-2)
        print(f"✅ M={M}, K={K}, N={N}, Triton and Torch match")
    except Exception as e:
        print(f"❌ M={M}, K={K}, N={N}, Triton and Torch differ", e)
        # print(torch_output)
        # print(triton_output)


    ms_torch = triton.testing.do_bench(lambda: torch.matmul(a, b))
    ms_triton = triton.testing.do_bench(lambda: matmul(a, b))
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    print(bcolors.WARNING + "torch:", "%.2f"%perf(ms_torch), "TFLOPS" + bcolors.ENDC)
    print(bcolors.WARNING + "triton:", "%.2f"%perf(ms_triton), "TFLOPS" + bcolors.ENDC)

    print(bcolors.OKBLUE + "-------------------------------------------------" + bcolors.ENDC)
    

for m in [4096,512,32,1]:
    for k,n in [(4096,4096),(4096,1024),(4096,14336),(14336,4096)]:
        benchmark(m,n,k)

# for m in [128 * i for i in range(2, 33)]:
#     n = m
#     k = m
#     benchmark(m,n,k)

# benchmark(512,512,512)