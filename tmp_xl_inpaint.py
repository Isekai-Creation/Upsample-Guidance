import os

# SET env HF_HOME
os.environ["HF_HOME"] = "/dev/shm"
os.environ["TRANSFORMERS_CACHE"] = "/dev/shm"

from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor
import random
from sdxl_inpaint_upsample import StableDiffusionXLInpaintUpsamplingGuidancePipeline
import numpy as np
import torch
import torch_xla as xla
import time
import uuid
import argparse
import requests
from io import BytesIO
from vision_process import get_image, get_prompt
from PIL import Image
from zipfile import ZipFile


def main(
    model,
    prompt,
    seed,
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
    dtype = torch.bfloat16 if num_images_per_prompt >= 64 else torch.float32

    # if no prompt is provided, generate a random prompt
    if not prompt:
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_url,
                    },
                    {"type": "text", "text": "Describe this image. In detail."},
                ],
            }
        ]
        prompt = get_prompt(message)

    # Load the pipeline
    pipeline = StableDiffusionXLInpaintUpsamplingGuidancePipeline.from_pretrained(
        model, torch_dtype=dtype
    ).to(device)

    init_image = None
    if image_url:
        init_image = get_image(image_url, resized_width=width, resized_height=height)

    start_generation = time.time()
    if seed == -1:
        seed = random.randint(0, 2147483647)
    print(f"Generated Seed: {seed}")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Generate the image and log metrics
    images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        image=init_image,  # Use the initial image for image-to-image generation
        strength=1,
        guidance_scale=7.5,
        guidance_rescale=0.7,
        num_images_per_prompt=num_images_per_prompt,
        height=height,
        width=width,
        generator=generator,
    ).images

    time_taken_generation = time.time() - start_generation
    print(f"Time Taken for Generation: {time_taken_generation}")

    time_taken = time.time() - start
    print(f"Total Time: {time_taken}")

    if ext_ip:
        # Send a POST request with all the images to ext_ip
        # if there are more than 8 images, send the first 8 images
        # the rest of the images will be sent as a zip file
        files = []
        for img in images[:8]:
            img_bytes = BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            files.append(
                ("files", (f"image_{uuid.uuid4()}.png", img_bytes, "image/png"))
            )

        # If there are more than 8 images, prepare the remaining images as a zip file
        if num_images_per_prompt > 8:
            zip_bytes = BytesIO()
            with ZipFile(zip_bytes, "w") as zip_file:
                for idx, img in enumerate(images):  # Images from 9th onwards
                    img_bytes = BytesIO()
                    img.save(img_bytes, format="PNG")
                    img_bytes.seek(0)
                    zip_file.writestr(f"image_{idx+1}.png", img_bytes.read())
            zip_bytes.seek(0)
            files.append(
                (
                    "files",
                    (
                        f"{uuid.uuid4()}_{seed}.zip",
                        zip_bytes,
                        "application/zip",
                    ),
                )
            )

        print(f"Sending POST request to {SERVER_URL} from {ext_ip}")
        data = {
            "ip": ext_ip,
            "model": model,
            "seed": seed,
            "prompt": prompt,
            "nsfw": False,
            "count": num_images_per_prompt,
            "zip": num_images_per_prompt > 8,
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
    elif save_dir != "~":
        save_dir = os.path.expanduser(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        print("Length Images: ", len(images))
        for img in images:
            # Save the image
            unique_id = str(uuid.uuid4())
            save_path = os.path.join(save_dir, f"test_{unique_id}.png")
            print(f"Saving Image: {save_path}")
            img.save(save_path)
    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion Image Generation")
    parser.add_argument(
        "--model",
        type=str,
        default="KBlueLeaf/Kohaku-XL-Zeta",
        help="HuggingFace Model name",
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt for image generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Seed for random number generation",
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
        "--num_images_per_prompt",
        type=int,
        default=8,
        help="Number of images to generate",
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
        help="Width of the generated image",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height of the generated image",
    )
    parser.add_argument(
        "--scale_factor",
        type=int,
        default=1,
        help="Scale factor for the generated image",
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
        help="Server URL to send the images",
    )

    args = parser.parse_args()
    main(
        args.model,
        args.prompt,
        args.seed,
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
