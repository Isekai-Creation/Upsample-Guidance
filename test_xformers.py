import torch
from xformers.components.attention import ScaledDotProduct
import torch_xla as xla
import torch_xla.core.xla_model as xm


device = xm.xla_device()

attention = ScaledDotProduct().cuda()

# FW a random bunch of data
inputs = torch.rand((16, 1024, 1024), device=device)

# Not a very sparse mask to begin with
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

mask = (torch.rand((1024, 1024)) < 0.9).cuda()
att = attention(q=inputs, k=inputs, v=inputs, mask=mask)

torch.cuda.synchronize()
max_memory = torch.cuda.max_memory_allocated() // 2 ** 20
print(f"Dense - Peak memory use: {max_memory}MB")

# Now use a very sparse mask and observe that memory use changes
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

mask = (torch.rand((1024, 1024)) < 0.1).cuda()
att = attention(q=inputs, k=inputs, v=inputs, mask=mask)

torch.cuda.synchronize()
max_memory = torch.cuda.max_memory_allocated() // 2 ** 20
print(f"Sparse - Peak memory use: {max_memory}MB")
