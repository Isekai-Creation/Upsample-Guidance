from sd15_upsample import StableDiffusionUpsamplingGuidancePipeline
import torch
import torch_xla as xla
import torch_xla.core.xla_model as xm
from diffusers import DiffusionPipeline
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_xla.distributed.xla_backend
import datetime
import time
import uuid
start = time.time()

# Load the pipeline
pipeline = DiffusionPipeline.from_pretrained("KBlueLeaf/Kohaku-XL-Epsilon-rev3", device=xm.xla_device())
# pipeline = StableDiffusionUpsamplingGuidancePipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", device=xm.xla_device())

prompt = "1girl,[hyuuga azuri,torino aqua | kamo kamen,mamyouda], solo,split ponytail,clothes_theft,glowing,looking at viewer,bubble,water, <lora:stareye starlight_XL_bx-v1.02:1>,blue stareye, masterpiece,best quality,great quality,newest,recent,absurdres"
negative_prompt = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"

# Generate the image and log metrics
images = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    num_images_per_prompt=4,
    height=1024,
    width=1024,
    time_factor=1.2,
    scale_factor=2,
    us_eta=0.6,
).images

print("Length Images: ", len(images))
for img in images:
    # Save the image
    unique_id = str(uuid.uuid4())
    print(f"Saving Image: test/test_{unique_id}.png")
    img.save(f"test/test_{unique_id}.png")

print(f"Total Time: {time.time() - start}")
