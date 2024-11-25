from __future__ import annotations
import os
import base64
import math
from io import BytesIO

import requests
import torch
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, StaticCache
import time
from vision_process import process_vision_info


try:
    import numpy as np
    import torch_xla.core.xla_model as xm
    import torch_xla as xla
    from torch_xla import runtime as xr
    import torch_xla.distributed.spmd as xs
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
        _prepare_spmd_partition_spec,
        SpmdFullyShardedDataParallel as FSDPv2,
    )

    xr.initialize_cache("/tmp")

    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices // 1, 1)
    device_ids = np.array(range(num_devices))
    # To be noted, the mesh must have an axis named 'fsdp', which the weights and activations will be sharded on.
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
    xs.set_global_mesh(mesh)

    print("_________________________XLA is Available!")
    XLA_AVAILABLE = True
except:
    print("_________________________XLA is not installed.")
    XLA_AVAILABLE = False

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


# set env TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD
os.environ["TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD"] = "99999999999999999999"


def get_prompt(message_obj):
    device = xla.device()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    )
    model = model.to(device)
    model = FSDPv2(model)
    model.eval()

    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4", max_pixels=max_pixels
    )

    texts = [
        processor.apply_chat_template(
            message_obj, tokenize=False, add_generation_prompt=True
        )
    ]
    image_inputs, video_inputs = process_vision_info(message_obj)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    xs.mark_sharding(
        inputs["input_ids"],
        xs.get_global_mesh(),
        _prepare_spmd_partition_spec(inputs["input_ids"]),
    )
    xs.mark_sharding(
        inputs["attention_mask"],
        xs.get_global_mesh(),
        _prepare_spmd_partition_spec(inputs["attention_mask"]),
    )
    batch_size, sequence_length = inputs["input_ids"].shape
    max_cache_length = 1024

    # setup static cache
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_length,
        device=model.device,
        dtype=model.dtype,
    )

    cache_position = torch.arange(sequence_length, device=device)

    attention_mask = inputs["attention_mask"]
    pos_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
    # Inference
    generated_ids = model(
        **inputs,
        cache_position=cache_position,
        return_dict=False,
        use_cache=True,
        position_ids=pos_ids,
        past_key_values=past_key_values,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    xla.sync()
    return output_text[0]


if __name__ == "__main__":
    start = time.time()
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://cdn.discordapp.com/attachments/1305813706168336394/1309027258660425738/image_7c137ffa-2fc0-40b3-9507-b70b705e2e6c.png?ex=67440aa3&is=6742b923&hm=492d749c741f50de075bebb7bb43819677fd30e5f54562f7f0fe92034fe8ca53&",
                },
                {"type": "text", "text": "Describe this image. In detail."},
            ],
        }
    ]
    prompt = get_prompt(message)

    print(f"Prompt: {prompt}")

    print(f"Time taken: {time.time() - start}")
