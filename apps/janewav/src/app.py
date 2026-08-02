import os
from pathlib import Path

from PIL import Image
import numpy
from numpy.typing import NDArray

from src.screen_capture import ScreenCapture
from src.image_reader import ImageReader
from src.audo_speaker import AudioSpeaker
from src.dependencies import get_settings
from pynput import keyboard, mouse


class App:
    def __init__(self):
        self.settings = get_settings()

        self.image_reader = ImageReader()
        self.audio_speaker = AudioSpeaker()

    def load_existing_image(self) -> NDArray | None:
        image_path = Path(f"{self.settings.script_path}/../bottom_left_corner.png")

        if image_path.exists():
            with Image.open(image_path) as image_data:
                return numpy.array(image_data)
            return image

        return None

    def on_click(self, x, y, button, pressed):
        if button == mouse.Button.right and pressed:
            print(f"Clicked at ({x}, {y})")

            if not self.settings.is_debug:
                screen_reader = ScreenCapture(get_settings().window_title)
                image = screen_reader.capture_screen()
            else:
                image = self.load_existing_image()

            # TODO: Determine if we can interrupt and gradually capture... although if the text is too short the game just swoops past the dialog and we never see it.
            if (
                image is not None
                and (speaker := self.image_reader.get_speaker_text(image)) is not None
            ):
                if speaker.name in self.settings.model_paths:
                    self.audio_speaker.speak_text(speaker.name, speaker.dialog)
                else:
                    print(f"Speaker name not found for: {speaker.name}")

    def force_quit(self):
        print("\nKill switch activated! Shutting down...")
        os._exit(0)

    def run_program(self):

        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.mouse_listener.start()

        print("Watching for text")

        with keyboard.GlobalHotKeys(
            {self.settings.terminate_command: self.force_quit}
        ) as hotkey_listener:
            # This blocks the main thread, waiting specifically for your secret combo
            hotkey_listener.join()
