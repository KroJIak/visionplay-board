"""
Base game class for VisionPlay Board games.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class BaseGame(ABC):
    """Abstract base class for all games."""
    
    def __init__(self, name: str, config):
        """Initialize base game."""
        self.name = name
        self.config = config
        self.is_active = False
        self.score = 0
        self.start_time = None
    
    @abstractmethod
    def start(self):
        """Start the game."""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop the game."""
        pass
    
    @abstractmethod
    def update(self, frame: np.ndarray, pose_results: Optional[Dict[str, Any]]) -> np.ndarray:
        """Update game state and return modified frame."""
        pass
    
    def get_score(self) -> int:
        """Get current game score."""
        return self.score
    
    def is_running(self) -> bool:
        """Check if game is currently running."""
        return self.is_active
