from abc import ABC, abstractmethod
from pathlib import Path

from .typing import FrameType


class Transformer(ABC):
    @abstractmethod
    def transform(self, X: FrameType) -> FrameType: ...

    def fit(self, X: FrameType, y: FrameType | None = None):
        pass

    def fit_transform(self, X: FrameType, y: FrameType | None = None) -> FrameType:
        self.fit(X, y)
        return self.transform(X)

    @property
    def log_dir(self) -> Path | None:
        try:
            return self._log_dir
        except AttributeError:
            self._log_dir: Path | None = None
            return self._log_dir

    @log_dir.setter
    def log_dir(self, value: Path):
        self._log_dir = value
