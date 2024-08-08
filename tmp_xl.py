from sdxl_upsample import StableDiffusionXLUpsamplingGuidancePipeline
import torch
import torch_xla as xla
import torch_xla.core.xla_model as xm
from diffusers import DiffusionPipeline
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_xla.distributed.xla_backend
import datetime



# Load the pipeline
pipeline = StableDiffusionXLUpsamplingGuidancePipeline.from_pretrained("KBlueLeaf/Kohaku-XL-Epsilon-rev3", device=xm.xla_device())


prompt = "1girl,[hyuuga azuri, mamyouda, sy4 | kamo kamen, popman3580, aiko \(kanl\), torino aqua | kurasawa moko, maccha \(mochancc\)], solo,Hair that transitions into a puzzle,striped_pants,upper body,looking at viewer, masterpiece,best quality,great quality,newest,recent,absurdres"
negative_prompt = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"

# Generate the image and log metrics
images = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=1024 *2,
    width=1024 *2,
    time_factor=1.2,
    scale_factor=2,
    us_eta=0.6,
).images

print("Length Images: ", len(images))

img = images[0]
# Save the image
img.save("test_us.png")