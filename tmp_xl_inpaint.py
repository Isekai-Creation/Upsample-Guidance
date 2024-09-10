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
import os
from vision_process import get_image
from PIL import Image


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

    # Load the safety checker and feature extractor
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    ).to(device)
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    # Function to check for NSFW content and return a black image if found
    def check_nsfw_images(
        images: list[Image.Image], output_type: str | None = "pil"
    ) -> list[Image.Image]:
        safety_checker_input = feature_extractor(images, return_tensors="pt").to(device)
        images_np = [np.array(img) for img in images]

        _, has_nsfw_concepts = safety_checker(
            images=images_np,
            clip_input=safety_checker_input.pixel_values.to(device),
        )

        # Replace NSFW images with black images
        for i, nsfw in enumerate(has_nsfw_concepts):
            if nsfw:
                print(
                    f"NSFW content detected in image {i}, replacing with black image."
                )
                # Create a black image of the same size
                black_image = Image.new("RGB", images[i].size, color=(0, 0, 0))
                images[i] = black_image

        return images, any(has_nsfw_concepts)

    # Load the pipeline
    pipeline = StableDiffusionXLInpaintUpsamplingGuidancePipeline.from_pretrained(
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
        strength=1,
        guidance_scale=7.5,
        guidance_rescale=0.7,
        num_images_per_prompt=num_images_per_prompt,
        height=height,
        width=width,
        generator=generator,
    ).images

    time_taken = time.time() - start
    print(f"Total Time: {time_taken}")

    del pipeline

    # Check for NSFW content and replace with black images if necessary
    images, has_nsfw_concepts = check_nsfw_images(images)

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
            "nsfw": has_nsfw_concepts,
        }
        response = requests.post(f"{SERVER_URL}/image-done", files=files, data=data)
        print(
            f"POST request sent to {SERVER_URL} from {ext_ip}, response status: {response.status_code}"
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
