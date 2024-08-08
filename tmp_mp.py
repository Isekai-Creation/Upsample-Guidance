from diffusers import DiffusionPipeline
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_backend
import torch_xla.distributed.xla_multiprocessing as xmp
import time
import os
import uuid
# Function to be executed on each process
def _mp_fn(index):
    try:
        device = xm.xla_device()
        print(f"INITIAL DEVICE: Process {index} Device: {device}")
        devices = xm.get_xla_supported_devices()
        print(f"Devices: {devices}; Total Devices: {len(devices)}")
        device = devices[index]
        print(f"Process {index} Device: {device}")

        # Load the pipeline on the current XLA device
        pipeline = DiffusionPipeline.from_pretrained("KBlueLeaf/Kohaku-XL-Epsilon-rev3", device=device)

        prompt = "1girl,[hyuuga azuri,torino aqua | kamo kamen,mamyouda], solo,split ponytail,clothes_theft,glowing,looking at viewer,bubble,water, <lora:stareye starlight_XL_bx-v1.02:1>,blue stareye, masterpiece,best quality,great quality,newest,recent,absurdres"
        negative_prompt = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"

        # Generate the image and log metrics
        images = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=4,
            num_inference_steps=25,
            height=1024,
            width=1024,
            time_factor=1.2,
            scale_factor=2,
            us_eta=0.6,
        ).images

        print(f"Process {index} Done!")

        for img in images:
            # Save the image
            unique_id = str(uuid.uuid4())
            print(f"Saving Image: test/test_{unique_id}.png")
            img.save(f"test/test_{index}_{unique_id}.png")

    except Exception as e:
        print(f"Error in Process {index}: {e}")
        pass

if __name__ == '__main__':


    start = time.time()
    xmp.spawn(_mp_fn, args=(),nprocs=8)

    print(f"Total Time: {time.time() - start}")
