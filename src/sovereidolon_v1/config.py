from __future__ import annotations

from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    def seed_for(self, run_id: str) -> int:
        return abs(hash((run_id, self.open_seed))) % (2**32)
