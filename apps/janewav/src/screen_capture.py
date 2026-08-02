import ctypes
from typing import List

from PIL import ImageGrab
from numpy.typing import NDArray
import pygetwindow
import numpy
from src.dependencies import get_settings
from src.settings import Settings


class ScreenCapture:
    _window_title: str
    settings: Settings = get_settings()

    def __init__(self, window_title: str):
        self._window_title = window_title

        try:
            # Set Process DPI Awareness to Per-Monitor Aware (Value 2)
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                # Fallback for older Windows systems
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    def capture_screen(self) -> NDArray | None:
        """_summary_

        Returns:
            NDArray | None: _description_
        """
        windows: List[pygetwindow.Win32Window] = pygetwindow.getWindowsWithTitle(
            self._window_title
        )

        if windows != None:
            window = windows[0]

            game_window = {
                "left": window.left,
                "top": window.top,
                "width": window.width,
                "height": window.height,
            }

            # Set the size of the bounding box you want to capture (Width x Height)

            # Capture Bottom-Left Corner
            bottom_left_box = (
                int(
                    game_window["left"] + self.settings.box_offset_left
                ),  # X1 (Left edge)
                int(
                    game_window["top"]
                    + game_window["height"]
                    - self.settings.box_height
                    - self.settings.box_offset_bottom
                ),  # Y1 (Top of box)
                int(
                    game_window["left"]
                    + self.settings.box_width
                    + self.settings.box_offset_left
                ),  # X2 (Right of box)
                int(
                    game_window["top"]
                    + game_window["height"]
                    - self.settings.box_offset_bottom
                ),  # Y2 (Bottom edge)
            )

            bottom_left_image = ImageGrab.grab(bbox=bottom_left_box)
            left_image = numpy.array(bottom_left_image)

            if self.settings.is_debug:
                bottom_left_image.save("bottom_left_corner.png")

            return left_image

        return None
