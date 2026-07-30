"""Deployment pydantic structures (plan §3.3)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ContainerSpec(BaseModel):
    node_id: str
    role: Literal["standalone", "head", "worker"] = "standalone"
    image: str
    device_ids: list[str]
    gpu_indices: list[int]
    env: dict[str, str] = Field(default_factory=dict)
    ports: dict[str, int] = Field(default_factory=dict)   # {api: 8001, metrics: 8001}
    command: list[str] = Field(default_factory=list)
    shm_size: str = "16g"
    name: str = ""


class HealthPolicy(BaseModel):
    interval_s: float = 3.0
    timeout_s: float = 300.0
    retries: int = 2                # full PULLING->READY attempts before FAILED


class DeploymentSpec(BaseModel):
    model: str
    served_name: str
    engine_args: dict = Field(default_factory=dict)
    containers: list[ContainerSpec]
    health: HealthPolicy = Field(default_factory=HealthPolicy)


class Endpoints(BaseModel):
    openai_url: str
    metrics_url: str
    health_url: str
