import json
from pathlib import Path

from dotenv import load_dotenv
import os


class Settings:
    model_paths: dict[str, Path] = {}
    window_title: str
    script_path: Path
    box_offset_left: int
    box_offset_bottom: int
    box_width: int
    box_height: int
    use_cuda: bool
    is_debug: bool
    terminate_command: str

    def __init__(self):
        load_dotenv()

        raw_model_paths = os.getenv("MODELS", "{}")
        self.model_paths = json.loads(raw_model_paths)

        # 1. Paths to your files
        self.script_path = Path(__file__).resolve().parent
        for key, relative_path in self.model_paths.items():
            self.model_paths[key] = Path(f"{self.script_path}{relative_path}")

        self.window_title = os.getenv("GAME_WINDOW_TITLE", "Star Trek: Voyager")
        self.box_offset_left = int(os.getenv("BOX_OFFSET_LEFT", "350"))
        self.box_offset_bottom = int(os.getenv("BOX_OFFSET_BOTTOM", "80"))
        self.box_width = int(os.getenv("BOX_WIDTH", "1150"))
        self.box_height = int(os.getenv("BOX_HEIGHT", "380"))
        self.use_cuda = os.getenv("USE_CUDA", "True").lower() == "true"
        self.is_debug = os.getenv("IS_DEBUG", "False").lower() == "true"
        self.terminate_command = os.getenv(
            "TERMINATE_COMMAND", "<ctrl>+<alt>+<shift>+q"
        )
