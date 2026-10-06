"""Fail-closed GPU selection for the one-GPU rejected-trace pilot.

Slurm's CUDA_VISIBLE_DEVICES may be remapped by the job's device cgroup.  Never
use that local ordinal as an nvidia-smi physical index. This workstation's GRES
uses /dev/nvidia0..7; resolve SLURM_JOB_GPUS device minor to UUID via XML instead.
Kubernetes verifies a single exposed full-GPU UUID before setting CUDA visibility.
Some container runtimes inject devices without a UUID in NVIDIA_VISIBLE_DEVICES;
in that case, only an unambiguous single-device inventory is accepted. No torch
import is needed; the launcher's separate CUDA smoke validates compute access.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET


class GPUOccupiedError(RuntimeError):
    """Valid allocation, but its sampled occupancy exceeds the launch guard."""


def wait_gpu_runtime(*, wait_seconds=30, poll_seconds=2, **kwargs):
    """Bounded settling wait; never relax thresholds or retry invalid allocation."""
    if not 0 <= wait_seconds <= 60 or not 0 < poll_seconds <= 5:
        raise ValueError('Invalid bounded GPU settling interval')
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            return configure_gpu_runtime(**kwargs)
        except GPUOccupiedError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(poll_seconds, remaining))


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


def container_gpu_uuid(inherited):
    """Validate single-device exposure, with or without legacy UUID metadata."""
    if not os.environ.get("KUBERNETES_SERVICE_HOST") or not os.environ.get("DFLASH_POD_UID", "").strip():
        raise ValueError("--use-container-gpu requires Kubernetes and downward-API DFLASH_POD_UID")
    allocation = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    uuid_pattern = r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
    discover = allocation in (None, "", "void")
    if not discover and not re.fullmatch(uuid_pattern, allocation):
        raise ValueError("Unsupported NVIDIA_VISIBLE_DEVICES; require a UUID or inventory discovery, not all/none/ordinals/MIG")
    # An empty existing mask explicitly disables CUDA; never silently broaden it.
    # The sole logical ordinal is accepted only after the NVML inventory proves
    # that the container can access exactly this allocated device.
    if inherited not in (None, "0") and not (
        isinstance(inherited, str) and re.fullmatch(uuid_pattern, inherited)
        and (discover or inherited == allocation)
    ):
        raise ValueError("CUDA_VISIBLE_DEVICES must be unset, 0, or the allocated container GPU UUID")
    inventory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader",
    ], text=True, timeout=15)
    accessible = [line.strip() for line in inventory.splitlines() if line.strip()]
    if len(accessible) != 1 or not re.fullmatch(uuid_pattern, accessible[0]):
        raise RuntimeError("Container GPU inventory must expose exactly one full GPU UUID")
    if not discover and accessible != [allocation]:
        raise RuntimeError("Container accessible GPU inventory does not exactly match its single allocation")
    if inherited not in (None, "0", accessible[0]):
        raise ValueError("CUDA_VISIBLE_DEVICES does not match the single exposed GPU")
    return accessible[0]


def configure_gpu_runtime(gpu=None, use_visible_gpu=False, *, use_container_gpu=False, require_gpu=False,
                          max_memory_mib=1024, max_utilization=10):
    """Validate selection, check occupancy, then set/preserve CUDA visibility.

    Call once in the parent before models/CUDA are initialized. Spawned workers
    inherit the resulting environment. CPU mode does not query any GPU. The
    collector's read-only --preflight must return before calling this function.
    """
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if sum((gpu is not None, bool(use_visible_gpu), bool(use_container_gpu))) > 1:
        raise ValueError("Choose --gpu OR --use-visible-gpu OR --use-container-gpu, not both/multiple")
    if gpu is not None and job_id:
        raise ValueError("Physical --gpu override is forbidden inside Slurm; use --use-visible-gpu")
    if gpu is not None and os.environ.get("KUBERNETES_SERVICE_HOST"):
        raise ValueError("Physical --gpu override is forbidden inside Kubernetes; use --use-container-gpu")
    if use_container_gpu and job_id:
        raise ValueError("--use-container-gpu cannot be combined with a Slurm allocation")
    allocated_uuid = None
    if use_container_gpu:
        allocated_uuid = container_gpu_uuid(inherited)
        physical_gpu = None
        mode, visible = "kubernetes_container", allocated_uuid
    elif use_visible_gpu:
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
            raise ValueError("GPU execution requires --gpu, --use-visible-gpu, or --use-container-gpu")
        physical_gpu = None
        mode, visible = "cpu", ""
    provenance = {
        "mode": mode, "device": "cpu" if mode == "cpu" else "cuda:0",
        "physical_gpu_index": physical_gpu if mode == "manual" else None,
        "allocated_device_minor": physical_gpu if mode == "slurm_visible" else None,
        "inherited_cuda_visible_devices": inherited,
        "cuda_visible_devices": visible, "slurm_job_id": job_id,
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),
        "slurm_job_node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "container_allocated_gpu_uuid": allocated_uuid,
        "kubernetes_pod_uid": os.environ.get("DFLASH_POD_UID") if use_container_gpu else None,
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES") if use_container_gpu else None,
        "container_uuid_source": ("single_visible_inventory" if os.environ.get("NVIDIA_VISIBLE_DEVICES")
                                  in (None, "", "void") else "nvidia_env_uuid") if use_container_gpu else None,
    }
    if mode != "cpu":
        # Slurm physical /dev/nvidia minor -> UUID, NOT the remapped CUDA ordinal
        # or an assumed nvidia-smi index. Manual mode retains its NVML index API.
        query_id = allocated_uuid if use_container_gpu else (slurm_gpu_uuid(physical_gpu) if use_visible_gpu else str(physical_gpu))
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
            raise GPUOccupiedError(f"GPU {query_id} is occupied ({used} MiB, {utilization}%); no models loaded")
        provenance["occupancy_before_load"] = {"memory_mib": used, "utilization_percent": utilization}
        provenance["nvidia_smi_query_id"] = query_id
        provenance["nvidia_smi_index"] = nvml_index
    # Slurm visibility remains byte-for-byte intact. Kubernetes uses the verified
    # allocated UUID rather than assuming its NVML index is a CUDA ordinal.
    if not use_visible_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return provenance
