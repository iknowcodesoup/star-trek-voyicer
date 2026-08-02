from collections import namedtuple
import re

from PIL.Image import Image
from numpy.typing import NDArray
import pytesseract

from src.dependencies import get_settings

Speaker = namedtuple("Speaker", ["name", "dialog"])


class ImageReader:
    settings = get_settings()

    def __init__(self):

        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

    def post_text_cleanup(self, text: str) -> str:
        """_summary_

        Args:
            text (str): _description_

        Returns:
            str: _description_
        """
        text = re.sub(r"\|(?=\s|[a-zA-Z])", "I", text)

        return text

    def read_captured_image(self, image: NDArray) -> str:
        """_summary_

        Args:
            image (NDArray): _description_

        Returns:
            str: _description_
        """
        captured_text = self.post_text_cleanup(
            pytesseract.image_to_string(image).strip()
        )

        return captured_text

    def get_speaker_text(self, image: NDArray) -> Speaker | None:
        """_summary_

        Args:
            image (NDArray): _description_

        Returns:
            Speaker: _description_
        """
        image_text = self.read_captured_image(image)

        lines = image_text.splitlines()

        if lines:
            speaker_text = lines[0].strip()
            dialog = " ".join(lines[1:]).strip()

            speaker_fullname_array = speaker_text.split()
            speaker_name = (
                speaker_fullname_array[-1] if speaker_fullname_array else speaker_text
            )

            return Speaker(speaker_name, dialog)

        return None


""" 
        # Only trigger if new text appears and it's not empty
        if raw_text and raw_text != last_text:
            last_text = raw_text
            speaker, clean_text = get_speaker_and_text(raw_text)

            # Fetch corresponding voice ID
            voice_id = VOICE_MAPPING.get(speaker, VOICE_MAPPING["DEFAULT"])
            print(f"Speaking ({speaker}): {clean_text}")

            try:
                # Generate and play audio instantly
                pants = "bananas"
            except Exception as e:
                print(f"Audio Generation Error: {e}")
"""


"""
    def get_speaker_and_text(raw_text:):
        # Parses text to identify the speaker (e.g., 'JANEWAY: Status report.')
        match = re.match(r"^([A-Zs]+):s*(.*)", raw_text.strip())
        if match:
            speaker = match.group(1).strip()
            speech_text = match.group(2).strip()
            return speaker, speech_text
        return "DEFAULT", raw_text.strip()
"""
