from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor
import random
from sdxl_upsample import StableDiffusionXLUpsamplingGuidancePipeline
import numpy as np
import torch
import torch_xla as xla
import torch_xla.core.xla_model as xm
from diffusers import DiffusionPipeline
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_xla.distributed.xla_backend
from torch_xla import runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
    SpmdFullyShardedDataParallel as FSDPv2,
)

import time
import uuid
import argparse
from PIL import Image
import requests
import os
from io import BytesIO

from vision_process import get_image


def main(
    prompt,
    negative_prompt,
    num_inference_steps,
    num_images_per_prompt=1,
    image_url=None,
    save_dir="~",
    width=1024,
    height=1024,
    scale_factor=1,
    ext_ip=None,
    SERVER_URL=None,
):
    start = time.time()

    device = xla.device()

    # Load the pipeline
    pipeline = StableDiffusionXLUpsamplingGuidancePipeline.from_pretrained(
        "KBlueLeaf/Kohaku-XL-Zeta",
    ).to(device)

    init_image = None
    if image_url:
        init_image = get_image(image_url, resized_width=width, resized_height=height)

    seed = random.randint(0, 2147483647)
    print(f"Generated Seed: {seed}")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Generate the image and log metrics
    images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        image=init_image,  # Use the initial image for image-to-image generation
        num_images_per_prompt=num_images_per_prompt,
        height=height,
        width=width,
        generator=generator,
        time_factor=0.81,
        scale_factor=scale_factor,
        us_eta=0.49,
        guidance_scale=7.5,
        guidance_rescale=0.7,
    ).images

    time_taken = time.time() - start
    print(f"Total Time: {time_taken}")

    if ext_ip:
        # Send a POST request with all the images to ext_ip
        files = []
        for img in images:
            img_bytes = BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            files.append(
                ("files", (f"image_{uuid.uuid4()}.png", img_bytes, "image/png"))
            )
        print(f"Sending POST request to {SERVER_URL} from {ext_ip}")
        data = {
            "ip": ext_ip,
            "prompt": prompt,
            "nsfw": False,
        }  # Replace with the actual IP if needed
        response = requests.post(f"{SERVER_URL}/image-done", files=files, data=data)
        if response.ok:
            print(
                f"POST request sent to {SERVER_URL} from {ext_ip}, response status: {response.status_code}"
            )
        else:
            print(
                f"Failed to upload video. Server responded with status: {response.status_code} - {response.text}"
            )

    else:
        save_dir = os.path.expanduser(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        print("Length Images: ", len(images))
        for img in images:
            # Save the image
            unique_id = str(uuid.uuid4())
            save_path = os.path.join(save_dir, f"test_{unique_id}.png")
            print(f"Saving Image: {save_path}")
            img.save(save_path)

    # send a request to the server url /image-done endpoint, upload the images to the server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion Image Generation")
    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt for image generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
        help="Negative text prompt for image generation",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=25, help="Number of inference steps"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=8, help="Number of images"
    )

    parser.add_argument(
        "--image_url",
        type=str,
        default=None,
        help="URL of the initial image for image-to-image generation",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="~",
        help="Directory to save the generated images",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Directory to save the generated images",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Directory to save the generated images",
    )
    parser.add_argument(
        "--scale_factor",
        type=int,
        default=1,
        help="Directory to save the generated images",
    )
    parser.add_argument(
        "--ext_ip",
        type=str,
        default=None,
        help="External IP address of the server",
    )
    parser.add_argument(
        "--SERVER_URL",
        type=str,
        default=None,
        help="External IP address of the server",
    )

    args = parser.parse_args()
    main(
        args.prompt,
        args.negative_prompt,
        args.num_inference_steps,
        args.num_images_per_prompt,
        args.image_url,
        args.save_dir,
        args.width,
        args.height,
        args.scale_factor,
        args.ext_ip,
        args.SERVER_URL,
    )
