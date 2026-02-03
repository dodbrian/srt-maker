import whisper
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class Transcriber:
    MODEL_SIZES = [
        "tiny",
        "base",
        "small",
        "medium",
        "large",
        "large-v1",
        "large-v2",
        "large-v3",
    ]

    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if model_size not in self.MODEL_SIZES:
            raise ValueError(
                f"Invalid model size '{model_size}'. "
                f"Must be one of: {', '.join(self.MODEL_SIZES)}"
            )

        self.model_size = model_size
        self.language = language
        self.device = device
        self.model: Optional[Any] = None

    def load_model(self):
        logger.info(f"Loading Whisper model '{self.model_size}'...")
        self.model = whisper.load_model(self.model_size, device=self.device)
        logger.info(f"Model '{self.model_size}' loaded successfully")

    def transcribe(self, audio_path: str) -> Dict:
        if self.model is None:
            self.load_model()

        logger.info(f"Transcribing audio from {audio_path}")

        result = self.model.transcribe(
            audio_path, language=self.language, word_timestamps=False
        )

        detected_lang = result.get("language", "unknown")
        self.language = detected_lang

        logger.info(f"Transcription complete. Detected language: {detected_lang}")

        return result

    def get_segments(self, audio_path: str) -> List[Dict[str, Any]]:
        result = self.transcribe(audio_path)
        return result.get("segments", [])
