import torch
import torch_xla
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_gif

# import our module wrapper
from fme import FMEWrapper
import time
from vision_process import get_image


start = time.time()
try:
    # Load the pipeline with the appropriate dtype
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.bfloat16
    )
    pipe.to(torch_xla.device())  # Move to XLA device

    # Initialize chunk size and helper if needed
    chunk = 1
    print(
        "Total frames: ",
        pipe.unet.config.num_frames,
    )
    if chunk != 1:
        helper = FMEWrapper(
            num_temporal_chunk=chunk,
            num_spatial_chunk=chunk,
            num_frames=pipe.unet.config.num_frames,
        )
        helper.wrap(pipe)

    # Inference as normal
    image = get_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png",
        resized_width=1024,
        resized_height=576,
    )

    # Set the random seed for reproducibility
    generator = torch.manual_seed(42)

    # Perform inference with the pipeline
    frames = pipe(
        image, decode_chunk_size=7, generator=generator
    ).frames[0]

    # Export the generated frames to a GIF
    export_to_gif(frames, "generated_fme.gif", fps=7)
    
except Exception as e:
    print(e)
    raise e
finally:
    print("Time taken: ", time.time() - start)
    