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

    @staticmethod
    def detect_gpu() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except (ImportError, RuntimeError):
            return False

    def load_model(self):
        logger.info(f"Loading Whisper model '{self.model_size}'...")
        self.model = whisper.load_model(self.model_size, device=self.device)
        logger.info(f"Model '{self.model_size}' loaded successfully")

    def transcribe(self, audio_path: str) -> Dict:
        if self.model is None:
            self.load_model()

        logger.info(f"Transcribing audio from {audio_path}")
        if self.language:
            logger.info(f"Using specified language: {self.language}")

        # Reduce repetitions and hallucinations
        # temperature=0.0 for deterministic, consistent results
        # compression_ratio_threshold=2.4 filters out repetitive segments
        # condition_on_previous_text=False prevents context-based repetitions
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=True,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
        )

        detected_lang = result.get("language", "unknown")
        # Only update language if user didn't specify one
        if self.language is None:
            self.language = detected_lang
            logger.info(f"Transcription complete. Detected language: {detected_lang}")
        else:
            logger.info(
                f"Transcription complete. Used language: {self.language} (detected: {detected_lang})"
            )

        return result

    def get_segments(self, audio_path: str) -> List[Dict[str, Any]]:
        result = self.transcribe(audio_path)
        return result.get("segments", [])
