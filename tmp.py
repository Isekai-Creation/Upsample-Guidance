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
import argparse

def main(prompt, num_inference_steps):
    start = time.time()

    # Load the pipeline
    pipeline = DiffusionPipeline.from_pretrained("KBlueLeaf/Kohaku-XL-Epsilon-rev3", device=xm.xla_device())
    # pipeline = StableDiffusionUpsamplingGuidancePipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", device=xm.xla_device())

    negative_prompt = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"

    # Generate the image and log metrics
    images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion Image Generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps")

    args = parser.parse_args()
    main(args.prompt, args.num_inference_steps)
