import torch
import time 
from torch.utils.cpp_extension import load
from functools import partial
from typing import Optional

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(name='rms_norm_lib', 
           sources=['rms_norm.cu'], 
           extra_cuda_cflags=[
               "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math"
            ], 
           extra_cflags=['-std=c++17'])


# un-fused naive rms norm
def naive_rms_norm(x: torch.Tensor, g: float):
    # y'=x/rms(x) 1/rms(x) = rsqrtf(sum(x^2)/K)
    s_rms = torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True))
    y = (x * s_rms) * g
    return y


def run_benchmark(perf_func: callable, x: torch.Tensor, 
                  tag: str, out: Optional[torch.Tensor] = None, 
                  warmup: int = 10, iters: int = 1000,
                  show_all: bool = False):
    g = 1.0
    if out is not None: 
        out.fill_(0)      
    if out is not None:
        for i in range(warmup):
            perf_func(x, out, g)
    else:
        for i in range(warmup):
            _ = perf_func(x, g) 
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(x, out, g)
    else:
        for i in range(iters):
            out = perf_func(x, g) 
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>17}: {out_val}, time:{mean_time:.8f}ms")
    if show_all: print(out)
    return out, mean_time


print("-" * 85)
N, K = 4096, 512
print(" " * 40 + f"N={N}, K={K}")
x = torch.randn((N, K)).cuda().float().contiguous()
out = torch.zeros_like(x).cuda().float().contiguous()
run_benchmark(lib.rms_norm_f32,   x, "f32",   out)
run_benchmark(lib.rms_norm_f32x4, x, "f32x4", out)
run_benchmark(naive_rms_norm,     x, "f32_th")

print("-" * 85)
x_f16 = x.half()
out_f16 = out.half()
run_benchmark(lib.rms_norm_f16_f16,        x_f16, "f16f16",        out_f16)
run_benchmark(lib.rms_norm_f16_f32,        x_f16, "f16f32",        out_f16)
run_benchmark(lib.rms_norm_f16x2_f16,      x_f16, "f16x2f16",      out_f16)
run_benchmark(lib.rms_norm_f16x8_f16,      x_f16, "f16x8f16",      out_f16)
run_benchmark(lib.rms_norm_f16x8_f32,      x_f16, "f16x8f32",      out_f16)
run_benchmark(lib.rms_norm_f16x8_pack_f16, x_f16, "f16x8packf16",  out_f16)
run_benchmark(lib.rms_norm_f16x8_pack_f32, x_f16, "f16x8packf32",  out_f16)
run_benchmark(naive_rms_norm,              x_f16, "f16_th")
print("-" * 85)

print("-" * 85)
N, K = 4096, 1024
print(" " * 40 + f"N={N}, K={K}")
x = torch.randn((N, K)).cuda().float().contiguous()
out = torch.zeros_like(x).cuda().float().contiguous()
run_benchmark(lib.rms_norm_f32,   x, "f32",   out)
run_benchmark(lib.rms_norm_f32x4, x, "f32x4", out)
run_benchmark(naive_rms_norm,     x, "f32_th")

print("-" * 85)
x_f16 = x.half()
out_f16 = out.half()
run_benchmark(lib.rms_norm_f16_f16,        x_f16, "f16f16",        out_f16)
run_benchmark(lib.rms_norm_f16_f32,        x_f16, "f16f32",        out_f16)
run_benchmark(lib.rms_norm_f16x2_f16,      x_f16, "f16x2f16",      out_f16)
run_benchmark(lib.rms_norm_f16x8_f16,      x_f16, "f16x8f16",      out_f16)
run_benchmark(lib.rms_norm_f16x8_f32,      x_f16, "f16x8f32",      out_f16)
run_benchmark(lib.rms_norm_f16x8_pack_f16, x_f16, "f16x8packf16",  out_f16)
run_benchmark(lib.rms_norm_f16x8_pack_f32, x_f16, "f16x8packf32",  out_f16)
run_benchmark(naive_rms_norm,              x_f16, "f16_th")
print("-" * 85)

print("-" * 85)
N, K = 4096, 2048
print(" " * 40 + f"N={N}, K={K}")
x = torch.randn((N, K)).cuda().float().contiguous()
out = torch.zeros_like(x).cuda().float().contiguous()
run_benchmark(lib.rms_norm_f32x4, x, "f32x4", out)
run_benchmark(naive_rms_norm,     x, "f32_th")

print("-" * 85)
x_f16 = x.half()
out_f16 = out.half()
run_benchmark(lib.rms_norm_f16x2_f16,      x_f16, "f16x2f16",      out_f16)
run_benchmark(lib.rms_norm_f16x8_f16,      x_f16, "f16x8f16",      out_f16)
run_benchmark(lib.rms_norm_f16x8_f32,      x_f16, "f16x8f32",      out_f16)
run_benchmark(lib.rms_norm_f16x8_pack_f16, x_f16, "f16x8packf16",  out_f16)
run_benchmark(lib.rms_norm_f16x8_pack_f32, x_f16, "f16x8packf32",  out_f16)
run_benchmark(naive_rms_norm,              x_f16, "f16_th")
print("-" * 85)

print("-" * 85)
N, K = 4096, 4096
print(" " * 40 + f"N={N}, K={K}")
x = torch.randn((N, K)).cuda().float().contiguous()
out = torch.zeros_like(x).cuda().float().contiguous()
run_benchmark(lib.rms_norm_f32x4, x, "f32x4", out)
run_benchmark(naive_rms_norm,     x, "f32_th")

print("-" * 85)
x_f16 = x.half()
out_f16 = out.half()
run_benchmark(lib.rms_norm_f16x8_f16,      x_f16, "f16x8f16",      out_f16)
run_benchmark(lib.rms_norm_f16x8_f32,      x_f16, "f16x8f32",      out_f16)
run_benchmark(lib.rms_norm_f16x8_pack_f16, x_f16, "f16x8packf16",  out_f16)
run_benchmark(lib.rms_norm_f16x8_pack_f32, x_f16, "f16x8packf32",  out_f16)
run_benchmark(naive_rms_norm,              x_f16, "f16_th")
print("-" * 85)

print("-" * 85)
N, K = 4096, 8192
print(" " * 40 + f"N={N}, K={K}")
x_f16 = torch.randn((N, K)).cuda().half().contiguous()
out_f16 = torch.zeros_like(x_f16).cuda().half().contiguous()
run_benchmark(lib.rms_norm_f16x8_f16,      x_f16, "f16x8f16",      out_f16)
run_benchmark(lib.rms_norm_f16x8_f32,      x_f16, "f16x8f32",      out_f16)
run_benchmark(lib.rms_norm_f16x8_pack_f16, x_f16, "f16x8packf16",  out_f16)
run_benchmark(lib.rms_norm_f16x8_pack_f32, x_f16, "f16x8packf32",  out_f16)
run_benchmark(naive_rms_norm,              x_f16, "f16_th")
print("-" * 85)

print("-" * 85)
N, K = 8192, 8192
print(" " * 40 + f"N={N}, K={K}")
x_f16 = torch.randn((N, K)).cuda().half().contiguous()
out_f16 = torch.zeros_like(x_f16).cuda().half().contiguous()
run_benchmark(lib.rms_norm_f16x8_f16,      x_f16, "f16x8f16",      out_f16)
run_benchmark(lib.rms_norm_f16x8_f32,      x_f16, "f16x8f32",      out_f16)
run_benchmark(lib.rms_norm_f16x8_pack_f16, x_f16, "f16x8packf16",  out_f16)
run_benchmark(lib.rms_norm_f16x8_pack_f32, x_f16, "f16x8packf32",  out_f16)
run_benchmark(naive_rms_norm,              x_f16, "f16_th")
print("-" * 85)
