import threading

from piper import PiperVoice
import pyaudio
from src.dependencies import get_settings


class AudioSpeaker:
    voices = {}
    audio: pyaudio.PyAudio = pyaudio.PyAudio()
    settings = get_settings()

    def __init__(self):

        self.current_playback_id: int = 0
        self.playback_lock = threading.Lock()

        print("Loading models...")

        for model_name, model_path in self.settings.model_paths.items():
            self.voices[model_name] = PiperVoice.load(
                model_path, use_cuda=self.settings.use_cuda
            )

    def stop(self):
        """
        Instantly invalidates the current playback.
        Returns immediately. Does NOT block the UI.
        """
        self.current_playback_id += 1

    def speak_text(self, model_name: str, text: str):
        """
        Starts audio playback in a background daemon thread.
        """
        self.stop()

        active_id = self.current_playback_id

        thread = threading.Thread(
            target=self.speak_text_worker,
            args=(model_name, text, active_id),
            daemon=True,
        )

        thread.start()

    def speak_text_worker(self, model_name: str, text: str, playback_id: int):
        """
        Internal worker that handles the blocking stream generation and writing.
        """
        stream = None

        try:
            for chunk in self.voices[model_name].synthesize(text):
                if self.current_playback_id != playback_id:
                    break

                with self.playback_lock:
                    if self.current_playback_id != playback_id:
                        break

                    if stream is None:
                        stream = self.audio.open(
                            format=self.audio.get_format_from_width(chunk.sample_width),
                            channels=chunk.sample_channels,
                            rate=chunk.sample_rate,
                            output=True,
                        )

                    audio_bytes = chunk.audio_int16_bytes
                    frame_size = 2048

                    for i in range(0, len(audio_bytes), frame_size):
                        if self.current_playback_id != playback_id:
                            break
                        stream.write(audio_bytes[i : i + frame_size])

        finally:
            with self.playback_lock:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
