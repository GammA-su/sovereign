from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .utils import stable_hash


class AdmissionPolicy(BaseModel):
    required_lanes: List[str] = Field(
        default_factory=lambda: ["recompute", "consequence", "translation", "anchor"]
    )
    max_controller_overhead: float = 0.2
    reject_on_withheld_hits: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SOVEREIDOLON_")

    verify_budget_steps: int = 500
    break_budget_attempts: int = 200
    synth_budget: int = 50
    required_lanes: List[str] = Field(
        default_factory=lambda: ["recompute", "consequence", "translation", "anchor"]
    )
    kill_switch_threshold: float = 0.2
    open_seed: int = 1337
    sealed_seed: int = 4242
    policy_version: str = "v1"
    breaker_novelty_window: int = 5
    withheld_families: List[str] = Field(default_factory=lambda: ["scale", "permute"])
    controller_overhead_threshold: float = 0.2
    admission_policy: AdmissionPolicy = Field(default_factory=AdmissionPolicy)
    warm_start_store: Optional[str] = None
    promotion_store: Optional[str] = None
    prefer_promotion_store: bool = False
    prefer_promotion_tier: str = "sealed"
    promotion_tier_strict: bool = False
    retrieval_dataset: Optional[str] = None
    pyfunc_minimize_budget: int = 50
    is_sealed_run: bool = False

    def seed_for(self, run_id: str) -> int:
        digest = stable_hash({"run_id": run_id, "seed": self.open_seed})
        return int(digest[:8], 16)
