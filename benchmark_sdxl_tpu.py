from __future__ import annotations

import contextlib
import io
import json
import os
import re
import statistics
import time
import traceback
import warnings
from pathlib import Path
from typing import Any


MODEL_ID = "KBlueLeaf/Kohaku-XL-Zeta"
BACKEND = "torch_xla"
DEVICE_LABEL = "TPU v5e-8"
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 25
OUTPUT_TYPE = "pil"
COUNTS = (1, 8, 32, 128)
WARM_REPEATS = 3
SEED = 12345
PROMPT = "masterpiece, best quality, cinematic fantasy character portrait, detailed lighting"
NEGATIVE_PROMPT = (
    "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, "
    "extra digit, fewer digits, cropped, worst quality, low quality, normal quality, "
    "jpeg artifacts, signature, watermark, username, blurry, artist name"
)
RESULT_DIR = Path("_work/real_sdxl_torch_xla_benchmark")


def configure_env() -> None:
    os.environ.setdefault("HF_HOME", "/dev/shm/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/dev/shm/hf")
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
    warnings.filterwarnings("ignore")


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def write_summary(path: Path, report: dict[str, Any]) -> None:
    rows = report["results"]
    lines = [
        "# Real SDXL Torch/XLA TPU Benchmark",
        "",
        f"- Model: `{MODEL_ID}`",
        f"- Backend: `{BACKEND}`",
        f"- Device target: `{DEVICE_LABEL}`",
        f"- Resolution: `{WIDTH}x{HEIGHT}`",
        f"- Steps: `{NUM_INFERENCE_STEPS}`",
        f"- Output: full image `{OUTPUT_TYPE}`",
        f"- Prompt: `{PROMPT}`",
        "",
        "| num_images_per_prompt | status | cold latency sec | warm median sec | images/sec | XLA compile evidence | peak memory/error |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        metrics = row.get("xla_metrics", {})
        cold_metrics = row.get("xla_metrics_cold") or metrics.get("cold", {})
        warm_runs = row.get("xla_metrics_warm_runs") or metrics.get("warm_runs", [])
        first_warm = warm_runs[0] if warm_runs else {}
        evidence = (
            f"cold CompileTime samples={cold_metrics.get('CompileTime', {}).get('samples')}; "
            f"cold Uncached={cold_metrics.get('UncachedCompile', {}).get('value')}; "
            f"warm Cached={first_warm.get('CachedCompile', {}).get('value')}; "
            f"warm ExecuteReplicated={first_warm.get('ExecuteReplicated', {}).get('value')}"
        )
        raw_error = row.get("error") or row.get("peak_memory") or ""
        error = str(raw_error).splitlines()[0] if raw_error else ""
        cold = row.get("cold_latency_sec")
        warm = row.get("warm_median_latency_sec")
        ips = row.get("images_per_sec")
        lines.append(
            "| {n} | {status} | {cold} | {warm} | {ips} | {evidence} | {error} |".format(
                n=row["num_images_per_prompt"],
                status=row["status"],
                cold="" if cold is None else f"{cold:.4f}",
                warm="" if warm is None else f"{warm:.4f}",
                ips="" if ips is None else f"{ips:.4f}",
                evidence=evidence,
                error=str(error).replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_xla_metrics(report: str) -> dict[str, dict[str, Any]]:
    wanted = {
        "CompileTime",
        "UncachedCompile",
        "CachedCompile",
        "ExecuteTime",
        "ExecuteReplicated",
        "ExecuteComputation",
        "TransferFromDeviceTime",
        "TransferToDeviceTime",
        "DeviceLockWait",
        "InboundData",
        "OutboundData",
    }
    metrics: dict[str, dict[str, Any]] = {}
    current: str | None = None
    for line in report.splitlines():
        heading = re.match(r"^(Metric|Counter): (.+)$", line)
        if heading:
            current = heading.group(2)
            if current in wanted:
                metrics[current] = {"kind": heading.group(1)}
            continue
        if current not in wanted:
            continue
        total = re.search(r"TotalSamples: ([0-9]+)", line)
        value = re.search(r"Value: ([0-9]+)", line)
        accumulator = re.search(r"Accumulator: (.+)", line)
        if total:
            metrics[current]["samples"] = int(total.group(1))
        if value:
            metrics[current]["value"] = int(value.group(1))
        if accumulator:
            metrics[current]["accumulator"] = accumulator.group(1).strip()
    return metrics


def clear_xla_metrics(met: Any) -> None:
    if hasattr(met, "clear_all"):
        met.clear_all()


def sync_xla(xla: Any, xm: Any) -> None:
    with contextlib.suppress(Exception):
        xm.mark_step()
    xla.sync()


def get_memory_info(xm: Any, device: Any) -> dict[str, Any] | None:
    with contextlib.suppress(Exception):
        info = xm.get_memory_info(device)
        return dict(info) if isinstance(info, dict) else {"raw": str(info)}
    return None


def dtype_for_count(torch: Any, count: int) -> Any:
    # Mirrors tmp_xl.py production behavior in this repo.
    return torch.bfloat16 if count >= 64 else torch.float32


def config_name(module: Any) -> str | None:
    config = getattr(module, "config", None)
    return getattr(config, "_name_or_path", None) or getattr(module, "name_or_path", None)


def image_count(images: Any) -> int | None:
    try:
        return len(images)
    except TypeError:
        return None


def save_sample_image(images: Any, path: Path) -> str | None:
    if not images:
        return None
    first = images[0]
    if hasattr(first, "save"):
        first.save(path)
        return str(path)
    return None


def run_pipeline_call(
    pipe: Any,
    torch: Any,
    xla: Any,
    xm: Any,
    count: int,
    seed: int,
) -> tuple[float, Any]:
    started = time.perf_counter()
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_images_per_prompt=count,
        height=HEIGHT,
        width=WIDTH,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        time_factor=0.81,
        scale_factor=1,
        us_eta=0.49,
        guidance_scale=7.5,
        guidance_rescale=0.7,
        output_type=OUTPUT_TYPE,
    ).images
    sync_xla(xla, xm)
    return time.perf_counter() - started, result


def main() -> None:
    configure_env()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(RESULT_DIR / "raw.log")
    metrics_path = RESULT_DIR / "xla_metrics.txt"
    metrics_path.write_text("", encoding="utf-8")

    import diffusers
    import torch
    import torch_xla as xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.runtime as xr

    from sdxl_upsample import StableDiffusionXLUpsamplingGuidancePipeline

    device = xla.device()
    report: dict[str, Any] = {
        "model": MODEL_ID,
        "backend": BACKEND,
        "device": DEVICE_LABEL,
        "actual_xla_device": str(device),
        "height": HEIGHT,
        "width": WIDTH,
        "steps": NUM_INFERENCE_STEPS,
        "output_type": "image",
        "pipeline_output_type": OUTPUT_TYPE,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "counts": list(COUNTS),
        "warm_repeats": WARM_REPEATS,
        "torch_version": torch.__version__,
        "torch_xla_version": getattr(xla, "__version__", "unknown"),
        "diffusers_version": diffusers.__version__,
        "xla_runtime_device_count": xr.global_runtime_device_count(),
        "results": [],
    }

    logger.write("REAL_SDXL_TORCH_XLA_BENCHMARK_START")
    logger.write(f"FORCED_MODEL_ID={MODEL_ID}")
    logger.write(f"BACKEND={BACKEND}")
    logger.write(f"DEVICE_TARGET={DEVICE_LABEL}")
    logger.write(f"XLA_DEVICE={device}")
    logger.write(f"RESOLUTION={WIDTH}x{HEIGHT}")
    logger.write(f"STEPS={NUM_INFERENCE_STEPS}")
    logger.write(f"OUTPUT_TYPE={OUTPUT_TYPE} full image output")
    logger.write(f"PROMPT={PROMPT}")
    logger.write(f"COUNTS={COUNTS}")

    for count in COUNTS:
        dtype = dtype_for_count(torch, count)
        row: dict[str, Any] = {
            "model": MODEL_ID,
            "backend": BACKEND,
            "device": DEVICE_LABEL,
            "actual_xla_device": str(device),
            "height": HEIGHT,
            "width": WIDTH,
            "steps": NUM_INFERENCE_STEPS,
            "output_type": "image",
            "pipeline_output_type": OUTPUT_TYPE,
            "num_images_per_prompt": count,
            "dtype": str(dtype),
            "cold_latency_sec": None,
            "warm_latencies_sec": [],
            "warm_median_latency_sec": None,
            "images_per_sec": None,
            "xla_metrics": {},
            "xla_metrics_cold": {},
            "xla_metrics_warm_runs": [],
            "status": "failed",
            "error": None,
            "traceback": None,
            "peak_memory": None,
            "sample_image": None,
            "loaded_model_proof": {},
        }
        logger.write(f"COUNT_START num_images_per_prompt={count} dtype={dtype}")
        try:
            load_started = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                pipe = StableDiffusionXLUpsamplingGuidancePipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
            row["load_latency_sec"] = round(time.perf_counter() - load_started, 4)
            if hasattr(pipe, "set_progress_bar_config"):
                pipe.set_progress_bar_config(disable=True)

            proof = {
                "model_id": MODEL_ID,
                "pipeline_class": pipe.__class__.__name__,
                "unet_config_or_model_path": config_name(pipe.unet),
                "vae_config_or_model_path": config_name(pipe.vae),
                "scheduler": pipe.scheduler.__class__.__name__,
                "dtype": str(dtype),
                "xla_device": str(device),
                "resolution": f"{WIDTH}x{HEIGHT}",
                "steps": NUM_INFERENCE_STEPS,
                "prompt": PROMPT,
                "num_images_per_prompt": count,
            }
            row["loaded_model_proof"] = proof
            logger.write("MODEL_PROOF " + json.dumps(proof, sort_keys=True, default=json_default))

            clear_xla_metrics(met)
            cold_seconds, cold_images = run_pipeline_call(pipe, torch, xla, xm, count, SEED)
            cold_metrics_text = met.metrics_report()
            cold_metrics = parse_xla_metrics(cold_metrics_text)
            sample_path = RESULT_DIR / f"sample_n{count}.png"
            row["sample_image"] = save_sample_image(cold_images, sample_path)
            row["cold_latency_sec"] = round(cold_seconds, 4)
            row["cold_result_count"] = image_count(cold_images)
            row["xla_metrics_cold"] = cold_metrics
            row["peak_memory"] = get_memory_info(xm, device)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n\n===== count={count} cold =====\n")
                handle.write(cold_metrics_text)
            logger.write(
                f"COLD_DONE count={count} latency_sec={cold_seconds:.4f} "
                f"images={row['cold_result_count']} metrics={json.dumps(cold_metrics, sort_keys=True)}"
            )

            warm_latencies: list[float] = []
            warm_metrics: list[dict[str, Any]] = []
            for warm_idx in range(1, WARM_REPEATS + 1):
                clear_xla_metrics(met)
                warm_seconds, warm_images = run_pipeline_call(pipe, torch, xla, xm, count, SEED + warm_idx)
                warm_metrics_text = met.metrics_report()
                parsed_warm = parse_xla_metrics(warm_metrics_text)
                warm_latencies.append(round(warm_seconds, 4))
                warm_metrics.append(parsed_warm)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"\n\n===== count={count} warm={warm_idx} =====\n")
                    handle.write(warm_metrics_text)
                logger.write(
                    f"WARM_DONE count={count} run={warm_idx} latency_sec={warm_seconds:.4f} "
                    f"images={image_count(warm_images)} metrics={json.dumps(parsed_warm, sort_keys=True)}"
                )

            median_latency = statistics.median(warm_latencies)
            row["warm_latencies_sec"] = warm_latencies
            row["warm_median_latency_sec"] = round(median_latency, 4)
            row["images_per_sec"] = round(count / median_latency, 4)
            row["xla_metrics_warm_runs"] = warm_metrics
            row["xla_metrics"] = {
                "cold": cold_metrics,
                "warm_runs": warm_metrics,
            }
            row["status"] = "success"
            logger.write(
                f"COUNT_SUCCESS count={count} warm_median_sec={median_latency:.4f} "
                f"images_per_sec={count / median_latency:.4f}"
            )
            del pipe
            with contextlib.suppress(Exception):
                torch_xla_empty = getattr(xla, "empty_cache", None)
                if callable(torch_xla_empty):
                    torch_xla_empty()
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = type(exc).__name__ + ": " + str(exc)
            row["traceback"] = traceback.format_exc()
            error_path = RESULT_DIR / f"error_n{count}.txt"
            error_path.write_text(row["traceback"], encoding="utf-8")
            logger.write(f"COUNT_FAILED count={count} error={row['error']}")
            logger.write(row["traceback"])
        finally:
            report["results"].append(row)
            write_json(RESULT_DIR / "results.json", report)
            write_summary(RESULT_DIR / "summary.md", report)

    logger.write("REAL_SDXL_TORCH_XLA_BENCHMARK_DONE")
    print("REAL_SDXL_TORCH_XLA_BENCHMARK_RESULTS_JSON")
    print(json.dumps(report, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
