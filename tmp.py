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
from PIL import Image
import requests
from io import BytesIO
import os

def main(prompt, negative_prompt, num_inference_steps, image_url=None, save_dir="~"):
    start = time.time()

    # Load the pipeline
    pipeline = DiffusionPipeline.from_pretrained("KBlueLeaf/Kohaku-XL-Epsilon-rev3", device=xm.xla_device())
    # pipeline = StableDiffusionUpsamplingGuidancePipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", device=xm.xla_device())

    init_image = None
    if image_url:
        response = requests.get(image_url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((1024, 1024))

    # Generate the image and log metrics
    images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        image=init_image,  # Use the initial image for image-to-image generation
        num_images_per_prompt=4,
        height=1024,
        width=1024,
        time_factor=1.2,
        scale_factor=2,
        us_eta=0.6,
    ).images

    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("Length Images: ", len(images))
    for img in images:
        # Save the image
        unique_id = str(uuid.uuid4())
        save_path = os.path.join(save_dir, f"test_{unique_id}.png")
        print(f"Saving Image: {save_path}")
        img.save(save_path)

    print(f"Total Time: {time.time() - start}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion Image Generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name", help="Negative text prompt for image generation")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--image_url", type=str, help="URL of the initial image for image-to-image generation")
    parser.add_argument("--save_dir", type=str, default="~", help="Directory to save the generated images")

    args = parser.parse_args()
    main(args.prompt, args.negative_prompt, args.num_inference_steps, args.image_url, args.save_dir)
