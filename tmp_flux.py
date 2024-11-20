import os

# SET env HF_HOME
os.environ["HF_HOME"] = "/dev/shm/tmp_xl_models"


import torch
import torch_xla as xla
from diffusers import FluxPipeline
from flux_upsample import FluxUpsampleGuidancePipeline

device = xla.device()

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
)

pipe.enable_sequential_cpu_offload(
    device=device
)  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power


prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
image.save("flux-schnell.png")
