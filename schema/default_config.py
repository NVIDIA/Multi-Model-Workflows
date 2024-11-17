from dataclasses import (
    dataclass,
    field
)
from typing import Any, List, Union 

@dataclass
class NGCModel:
    """Single model instance."""

    name: str = "grounding_dino"
    version: str = "grounding_dino_swin_tiny_commercial_trainable_v1.0"
    org: str = "nvidia"
    team: str = "tao"
    entrypoint: str = "grounding_dino"
    local_model_path: Union[str, None] = None


@dataclass
class GradioApp:
    """Configuration of the gradio app."""

    server_name: str = "0.0.0.0"
    server_port: int = 8000  # default port to instantiate the app
    debug: bool = True
    max_file_size: Any = None


@dataclass
class NIMConfig:

    url: str = "https://integrate.api.nvidia.com/v1"
    api_key: str = "<API_KEY"



@dataclass
class GradioApp:
    """Configuration of the gradio app."""

    model: List[NGCModel] = field(
        default_factory=list,
        metadata={
            "description": ""
        }
    )
    app: GradioApp = GradioApp()
    nim: NIMConfig = NIMConfig()

