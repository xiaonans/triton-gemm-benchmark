import torch

# weight = [K,N]
def quantize(weight, bitwidth, groupsize=128, dequantize=False):
    assert weight.dim() == 2
    orig_shape = weight.shape
    K = orig_shape[0]
    N = orig_shape[1]

    assert K % groupsize == 0

    weight = weight.reshape(K//groupsize, groupsize, N).permute(1, 0, 2).reshape(groupsize, -1)
    num_groups = weight.shape[1]
    assert num_groups == N * (K // groupsize)

    max_val = weight.amax(dim=0, keepdim=True)
    min_val = weight.amin(dim=0, keepdim=True)

    assert max_val.numel() == num_groups

    max_int = (2**bitwidth-1) * torch.ones_like(max_val)
    min_int = torch.zeros_like(min_val)

    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val/scales))

    # import pdb;pdb.set_trace()
    weight = torch.clamp(torch.round(weight/scales) + zeros, min_int, max_int)

    if dequantize:
        output = (weight - zeros) * scales
        output = output.reshape(groupsize, -1, N).permute(1,0,2)
        return output.reshape(orig_shape).cuda()

    else:
        scales = scales.view(-1, N).cuda()
        zeros = zeros.view(-1, N).cuda()
        weight = weight.reshape(groupsize, -1, N).permute(1,0,2)
        output = weight.to(torch.uint8).reshape(orig_shape).cuda()
        return output, zeros, scales

