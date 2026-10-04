"""Fail-closed GPU selection for the one-GPU rejected-trace pilot.

Slurm's CUDA_VISIBLE_DEVICES may be remapped by the job's device cgroup.  Never
use that local ordinal as an nvidia-smi physical index. This workstation's GRES
uses /dev/nvidia0..7; resolve SLURM_JOB_GPUS device minor to UUID via XML instead.
No torch import is needed.
"""
from __future__ import annotations

import os
import re
import subprocess
import xml.etree.ElementTree as ET


def slurm_gpu_uuid(device_minor):
    """Resolve this host's GRES /dev/nvidiaN to UUID without assuming NVML order."""
    inventory = subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=15)
    try:
        root = ET.fromstring(inventory)
        matches = [gpu for gpu in root.findall("gpu")
                   if gpu.findtext("minor_number", "").strip() == str(device_minor)]
    except ET.ParseError as exc:
        raise RuntimeError("Cannot parse nvidia-smi device inventory; refusing to load models") from exc
    if len(matches) != 1:
        raise RuntimeError("Cannot uniquely map SLURM_JOB_GPUS device minor to nvidia-smi UUID")
    uuid = matches[0].findtext("uuid", "").strip()
    if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid):
        raise RuntimeError("Invalid nvidia-smi UUID for allocated device minor")
    return uuid


def configure_gpu_runtime(gpu=None, use_visible_gpu=False, *, require_gpu=False,
                          max_memory_mib=1024, max_utilization=10):
    """Validate selection, check occupancy, then set/preserve CUDA visibility.

    Call once in the parent before models/CUDA are initialized. Spawned workers
    inherit the resulting environment. CPU mode does not query any GPU. The
    collector's read-only --preflight must return before calling this function.
    """
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if gpu is not None and use_visible_gpu:
        raise ValueError("Choose --gpu OR --use-visible-gpu, not both")
    if gpu is not None and job_id:
        raise ValueError("Physical --gpu override is forbidden inside Slurm; use --use-visible-gpu")
    if use_visible_gpu:
        if not job_id or not re.fullmatch(r"[0-9]+", job_id):
            raise ValueError("--use-visible-gpu requires an active numeric SLURM_JOB_ID")
        allocation = os.environ.get("SLURM_JOB_GPUS", "")
        if not re.fullmatch(r"[0-7]", allocation):
            raise ValueError("Expected exactly one physical GPU 0..7 in SLURM_JOB_GPUS")
        if inherited is None or not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", inherited):
            raise ValueError("Expected exactly one inherited CUDA_VISIBLE_DEVICES index or GPU UUID")
        physical_gpu = int(allocation)
        mode, visible = "slurm_visible", inherited
    elif gpu is not None:
        if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
            raise ValueError("--gpu must be a nonnegative physical index")
        physical_gpu = gpu
        mode, visible = "manual", str(gpu)
    else:
        if require_gpu:
            raise ValueError("GPU execution requires --gpu or --use-visible-gpu")
        physical_gpu = None
        mode, visible = "cpu", ""
    provenance = {
        "mode": mode, "device": "cpu" if physical_gpu is None else "cuda:0",
        "physical_gpu_index": physical_gpu if mode == "manual" else None,
        "allocated_device_minor": physical_gpu if mode == "slurm_visible" else None,
        "inherited_cuda_visible_devices": inherited,
        "cuda_visible_devices": visible, "slurm_job_id": job_id,
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),
        "slurm_job_node_list": os.environ.get("SLURM_JOB_NODELIST"),
    }
    if physical_gpu is not None:
        # Slurm physical /dev/nvidia minor -> UUID, NOT the remapped CUDA ordinal
        # or an assumed nvidia-smi index. Manual mode retains its NVML index API.
        query_id = slurm_gpu_uuid(physical_gpu) if use_visible_gpu else str(physical_gpu)
        result = subprocess.check_output([
            "nvidia-smi", f"--id={query_id}",
            "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits",
        ], text=True, timeout=15)
        try:
            nvml_index, used, utilization = (int(part.strip()) for part in result.strip().split(","))
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Cannot establish GPU occupancy from nvidia-smi; refusing to load models") from exc
        if nvml_index < 0 or used < 0 or not 0 <= utilization <= 100:
            raise RuntimeError("Invalid GPU occupancy from nvidia-smi; refusing to load models")
        if used > max_memory_mib or (max_utilization is not None and utilization > max_utilization):
            raise RuntimeError(f"GPU {physical_gpu} is occupied ({used} MiB, {utilization}%); no models loaded")
        provenance["occupancy_before_load"] = {"memory_mib": used, "utilization_percent": utilization}
        provenance["nvidia_smi_query_id"] = query_id
        provenance["nvidia_smi_index"] = nvml_index
    # In inherited mode leave the scheduler's environment byte-for-byte intact.
    if not use_visible_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return provenance
