"""Validated configuration; unknown keys are mistakes, not silent defaults."""

from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "xgboost"
    version: str = "1.0.0"
    params: dict = Field(default_factory=lambda: {"n_estimators": 40, "max_depth": 3})


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int = Field(32, gt=0)
    learning_rate: float = Field(0.01, gt=0)
    epochs: int = Field(50, gt=0)
    device: str = "cpu"
    seed: int = 42
    test_size: float = Field(0.2, gt=0, lt=1)


class FairnessConfig(BaseModel):
    enabled: bool = True
    metrics: list[str] = Field(default_factory=lambda: ["demographic_parity", "equalized_odds"])
    protected_attributes: list[str] = Field(default_factory=list)
    threshold: float = Field(0.1, ge=0, le=1)


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PEARLMIND_", env_nested_delimiter="__", extra="forbid"
    )
    project_name: str = "PearlMind ML Journey"
    version: str = "2.0.0"
    environment: str = "development"
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    fairness: FairnessConfig = Field(default_factory=FairnessConfig)

    @classmethod
    def from_yaml(cls, path):
        with Path(path).open() as f:
            return cls(**(yaml.safe_load(f) or {}))

    def save_yaml(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.model_dump()))
