from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    """
    Runtime settings for serving.

    We keep this intentionally small:
    - artifacts_dir: where joblib models live (expects artifacts/v1/{in,us}/*.joblib)
    """

    artifacts_dir: Path = Path("artifacts")

    def model_dir(self, market: str) -> Path:
        return self.artifacts_dir / "v1" / market.lower()
