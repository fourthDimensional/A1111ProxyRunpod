import os
import json
import time
import random
import logging
import requests
import uvicorn
from logging.handlers import RotatingFileHandler
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# ==============================================================
# Load environment variables from .env file
# ==============================================================
load_dotenv()

# Configuration from .env
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
COST_PER_SECOND = float(os.getenv("COST_PER_SECOND", "0.00031"))
RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

# ==============================================================
# Logging setup
# ==============================================================
LOG_FILE = os.getenv("LOG_FILE", "proxy.log")

logger = logging.getLogger("proxy")
logger.setLevel(logging.INFO)

# Rotating log handler with compression-like rollover (just rotates)
handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ==============================================================
# FastAPI application
# ==============================================================
app = FastAPI(title="ComfyUI to A1111 API Proxy", version="1.0.0")


# ==============================================================
# Models
# ==============================================================
class Txt2ImgRequest(BaseModel):
    """Schema for A1111-compatible txt2img request."""
    prompt: str
    negative_prompt: Optional[str] = ""
    steps: Optional[int] = 25
    cfg_scale: Optional[float] = 7.0
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    seed: Optional[int] = -1
    batch_size: Optional[int] = 1
    sampler_name: Optional[str] = "Euler"
    scheduler: Optional[str] = "normal"
    enable_hr: Optional[bool] = False
    denoising_strength: Optional[float] = 0
    hr_scale: Optional[float] = 2.0
    hr_upscaler: Optional[str] = None
    hr_second_pass_steps: Optional[int] = 0
    n_iter: Optional[int] = 1
    restore_faces: Optional[bool] = False
    tiling: Optional[bool] = False


# ==============================================================
# ComfyUI Workflow Template (static base structure)
# ==============================================================
WORKFLOW_TEMPLATE = {
    "4": {
        "inputs": {"ckpt_name": "noobaicyberfix_v30.safetensors"},
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": "Load Checkpoint - BASE"}
    },
    "5": {
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        "class_type": "EmptyLatentImage",
        "_meta": {"title": "Empty Latent Image"}
    },
    "6": {
        "inputs": {"text": "INSERT PROMPT HERE", "clip": ["4", 1]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"}
    },
    "7": {
        "inputs": {"text": "INSERT NEGATIVE PROMPT HERE", "clip": ["4", 1]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"}
    },
    "10": {
        "inputs": {
            "add_noise": "enable",
            "noise_seed": 158456888197893,
            "steps": 25,
            "cfg": 7,
            "sampler_name": "euler",
            "scheduler": "normal",
            "start_at_step": 0,
            "end_at_step": 25,
            "return_with_leftover_noise": "enable",
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0]
        },
        "class_type": "KSamplerAdvanced",
        "_meta": {"title": "KSampler (Advanced) - BASE"}
    },
    "17": {
        "inputs": {"samples": ["10", 0], "vae": ["4", 2]},
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"}
    },
    "19": {
        "inputs": {"filename_prefix": "ComfyUI", "images": ["17", 0]},
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"}
    }
}


# ==============================================================
# Utility functions
# ==============================================================
def create_workflow(request: Txt2ImgRequest) -> dict:
    """Create a ComfyUI workflow from A1111 request parameters."""
    workflow = json.loads(json.dumps(WORKFLOW_TEMPLATE))

    workflow["6"]["inputs"]["text"] = request.prompt
    workflow["7"]["inputs"]["text"] = request.negative_prompt
    workflow["5"]["inputs"]["width"] = request.width
    workflow["5"]["inputs"]["height"] = request.height
    workflow["5"]["inputs"]["batch_size"] = request.batch_size
    workflow["10"]["inputs"]["steps"] = request.steps
    workflow["10"]["inputs"]["cfg"] = request.cfg_scale
    workflow["10"]["inputs"]["sampler_name"] = request.sampler_name.lower()
    workflow["10"]["inputs"]["scheduler"] = "normal"
    workflow["10"]["inputs"]["end_at_step"] = request.steps

    if request.seed == -1:
        request.seed = random.randint(0, 2**32 - 1)
    workflow["10"]["inputs"]["noise_seed"] = request.seed

    logger.info(f"Workflow created for prompt: {request.prompt[:50]}")
    return workflow


def submit_runpod_job(workflow: dict) -> Optional[str]:
    """Submit job to RunPod and return job ID."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {RUNPOD_API_KEY}"}
    payload = {"input": {"workflow": workflow}}

    try:
        response = requests.post(f"{RUNPOD_BASE_URL}/run", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        job_id = result.get("id")
        logger.info(f"Job submitted: {job_id}, Status: {result.get('status')}")
        return job_id
    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        return None


def check_runpod_status(job_id: str) -> Optional[dict]:
    """Check RunPod job status."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {RUNPOD_API_KEY}"}
    try:
        response = requests.get(f"{RUNPOD_BASE_URL}/status/{job_id}", headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error checking status for job {job_id}: {e}")
        return None


def wait_for_job(job_id: str, max_wait: int = 1800) -> Optional[dict]:
    """Poll job until completion with cost tracking."""
    start = time.time()
    poll_interval = 1
    logger.info(f"Waiting for job {job_id}...")

    while time.time() - start < max_wait:
        status_response = check_runpod_status(job_id)
        if not status_response:
            return None

        status = status_response.get("status")
        elapsed = time.time() - start
        cost = elapsed * COST_PER_SECOND
        logger.info(f"Job {job_id} | Status: {status} | Elapsed: {int(elapsed)}s | Cost: ${cost:.4f}")

        if status == "COMPLETED":
            execution_time = status_response.get("executionTime", 0) / 1000
            total_cost = execution_time * COST_PER_SECOND
            logger.info(f"Job {job_id} completed in {execution_time:.2f}s | Total cost: ${total_cost:.6f}")
            return status_response
        elif status == "FAILED":
            logger.error(f"Job {job_id} failed: {status_response.get('error', 'Unknown error')}")
            return None

        time.sleep(poll_interval)

    logger.warning(f"Timeout: Job {job_id} did not complete within {max_wait} seconds")
    return None


def extract_image_base64(response: dict) -> Optional[str]:
    """Extract base64 image from RunPod response."""
    try:
        output = response.get("output", {})
        base64_data = None

        if "message" in output:
            base64_data = output["message"]
        elif "images" in output and len(output["images"]) > 0:
            base64_data = output["images"][0].get("data")

        if not base64_data:
            logger.error(f"Could not find image data in response: {json.dumps(output, indent=2)}")
            return None

        if "," in base64_data:
            _, encoded_data = base64_data.split(",", 1)
        else:
            encoded_data = base64_data
        return encoded_data
    except Exception as e:
        logger.error(f"Error extracting image: {e}")
        return None


# ==============================================================
# Routes (do not change routes or functionality)
# ==============================================================
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "ComfyUI to A1111 API Proxy",
        "status": "online",
        "docs": "/docs",
        "supported_endpoints": [
            "/sdapi/v1/txt2img", "/sdapi/v1/options",
            "/sdapi/v1/samplers", "/sdapi/v1/sd-models"
        ],
        "backend": {
            "type": "RunPod ComfyUI",
            "endpoint_id": RUNPOD_ENDPOINT_ID,
            "cost_per_second": COST_PER_SECOND
        }
    }


@app.post("/sdapi/v1/txt2img")
async def txt2img(request: Txt2ImgRequest):
    """A1111-compatible text-to-image endpoint."""
    try:
        logger.info(f"New request received. Prompt: {request.prompt[:60]}...")

        workflow = create_workflow(request)
        job_id = submit_runpod_job(workflow)
        if not job_id:
            raise HTTPException(status_code=500, detail="Failed to submit job to RunPod")

        result = wait_for_job(job_id)
        if not result:
            raise HTTPException(status_code=500, detail="Job failed or timed out")

        image_base64 = extract_image_base64(result)
        if not image_base64:
            raise HTTPException(status_code=500, detail="Failed to extract image from response")

        execution_time_ms = result.get("executionTime", 0)
        execution_time_s = execution_time_ms / 1000
        total_cost = execution_time_s * COST_PER_SECOND

        logger.info(f"Image generated successfully (Cost: ${total_cost:.6f})")

        return {
            "images": [image_base64],
            "parameters": request.model_dump(exclude_unset=True),
            "info": json.dumps({
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "steps": request.steps,
                "cfg_scale": request.cfg_scale,
                "width": request.width,
                "height": request.height,
                "seed": request.seed,
                "sampler_name": request.sampler_name,
                "runpod_job_id": job_id,
                "execution_time_ms": execution_time_ms,
                "execution_time_seconds": execution_time_s,
                "estimated_cost_usd": f"{total_cost:.6f}"
            })
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in txt2img endpoint")
        raise HTTPException(status_code=500, detail=str(e))


# Mocked endpoints for compatibility
@app.get("/sdapi/v1/options")
async def get_options():
    return {"sd_model_checkpoint": "noobaicyberfix_v30.safetensors", "sd_checkpoint_hash": "unknown"}


@app.get("/sdapi/v1/samplers")
async def get_samplers():
    return [{"name": "Euler", "aliases": ["euler"]}]


@app.get("/sdapi/v1/sd-models")
async def get_models():
    return [{"title": "noobaicyberfix_v30.safetensors", "model_name": "noobaicyberfix_v30"}]


@app.get("/internal/ping")
@app.get("/sdapi/v1/cmd-flags")
async def ping():
    return {"status": "ok"}


@app.get("/sdapi/v1/loras")
async def get_loras():
    return []


@app.get("/sdapi/v1/hypernetworks")
async def get_hypernetworks():
    return []


@app.get("/sdapi/v1/embeddings")
async def get_embeddings():
    return {"loaded": {}, "skipped": {}, "failed": {}}


@app.get("/sdapi/v1/face-restorers")
async def get_face_restorers():
    return []


@app.get("/sdapi/v1/upscalers")
async def get_upscalers():
    return []


@app.get("/sdapi/v1/progress")
async def get_progress():
    return {"progress": 0.0, "eta_relative": 0.0, "state": {}, "current_image": None}


# ==============================================================
# Entry point
# ==============================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("ComfyUI to A1111 API Proxy Server starting")
    logger.info("=" * 60)
    logger.info(f"RunPod Endpoint: {RUNPOD_ENDPOINT_ID}")
    logger.info(f"Cost per second: ${COST_PER_SECOND}")
    logger.info("Starting server on http://0.0.0.0:7860")
    logger.info("API docs available at http://localhost:7860/docs")
    logger.info("Compatible with: SillyTavern, Draw Things, and other A1111 clients")
    logger.info("=" * 60)

    uvicorn.run(app, host="127.0.0.1", port=7860)
