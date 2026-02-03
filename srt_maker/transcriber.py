import whisper
from typing import List, Dict, Optional, Any, Set
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


# Default thresholds for filtering hallucinated/low-quality segments
# These are tuned to catch obvious hallucinations while preserving quiet speech
DEFAULT_NO_SPEECH_THRESHOLD = 0.9  # Filter segments with no_speech_prob > this
DEFAULT_LOGPROB_THRESHOLD = -1.5  # Filter segments with avg_logprob < this
DEFAULT_MIN_DURATION = 0.1  # Minimum segment duration in seconds
DEFAULT_MAX_REPETITIONS = 2  # Max times same text can repeat consecutively
DEFAULT_SIMILARITY_THRESHOLD = (
    0.7  # Text similarity threshold for detecting similar hallucinations
)
DEFAULT_REPETITION_WINDOW = 60.0  # Time window (seconds) to look for similar text


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
        no_speech_threshold: float = DEFAULT_NO_SPEECH_THRESHOLD,
        logprob_threshold: float = DEFAULT_LOGPROB_THRESHOLD,
        min_segment_duration: float = DEFAULT_MIN_DURATION,
        max_repetitions: int = DEFAULT_MAX_REPETITIONS,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        repetition_window: float = DEFAULT_REPETITION_WINDOW,
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

        # Hallucination filtering thresholds
        self.no_speech_threshold = no_speech_threshold
        self.logprob_threshold = logprob_threshold
        self.min_segment_duration = min_segment_duration
        self.max_repetitions = max_repetitions
        self.similarity_threshold = similarity_threshold
        self.repetition_window = repetition_window

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

    @staticmethod
    def text_similarity(text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts (0.0 to 1.0)."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _find_similar_segments_in_window(
        self,
        segments: List[Dict[str, Any]],
        current_idx: int,
        text: str,
        start_time: float,
    ) -> int:
        """
        Count how many similar segments exist within the time window.
        Looks both backward and forward from current position.
        """
        count = 0
        for i, seg in enumerate(segments):
            if i == current_idx:
                continue
            seg_start = seg.get("start", 0.0)
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue
            # Check if within time window
            if abs(seg_start - start_time) <= self.repetition_window:
                similarity = self.text_similarity(text, seg_text)
                if similarity >= self.similarity_threshold:
                    count += 1
        return count

    def _detect_hallucination_clusters(
        self, segments: List[Dict[str, Any]]
    ) -> Set[int]:
        """
        Detect clusters of similar text that indicate hallucinations.
        Returns indices of segments that are part of hallucination clusters.
        """
        hallucination_indices: Set[int] = set()

        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            if not text:
                continue

            start_time = segment.get("start", 0.0)
            similar_count = self._find_similar_segments_in_window(
                segments, i, text, start_time
            )

            # If there are 3+ similar segments in the window, it's likely hallucination
            # Keep the first occurrence, mark others
            if similar_count >= 2:
                # Find all similar segments and keep only the first one
                first_occurrence_idx = None
                for j, seg in enumerate(segments):
                    seg_text = seg.get("text", "").strip()
                    seg_start = seg.get("start", 0.0)
                    if not seg_text:
                        continue
                    if abs(seg_start - start_time) <= self.repetition_window:
                        similarity = self.text_similarity(text, seg_text)
                        if similarity >= self.similarity_threshold:
                            if first_occurrence_idx is None:
                                first_occurrence_idx = j
                            elif j != first_occurrence_idx:
                                hallucination_indices.add(j)

        return hallucination_indices

    def filter_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out hallucinated and low-quality segments.

        Applies multiple filtering strategies:
        1. Remove segments with very high no_speech probability
        2. Remove segments with very low average log probability
        3. Remove very short segments (likely artifacts)
        4. Remove consecutive duplicate text (exact match)
        5. Remove hallucination clusters (similar text repeated in time window)
        """
        if not segments:
            return segments

        # First pass: detect hallucination clusters
        hallucination_indices = self._detect_hallucination_clusters(segments)

        filtered = []
        prev_text = None
        repetition_count = 0

        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            if not text:
                continue

            # Check if part of hallucination cluster
            if i in hallucination_indices:
                logger.debug(f"Filtered segment (hallucination cluster): {text[:50]}")
                continue

            # Check no_speech probability
            no_speech_prob = segment.get("no_speech_prob", 0.0)
            if no_speech_prob > self.no_speech_threshold:
                logger.debug(
                    f"Filtered segment (no_speech_prob={no_speech_prob:.2f}): {text[:50]}"
                )
                continue

            # Check average log probability (confidence)
            avg_logprob = segment.get("avg_logprob", 0.0)
            if avg_logprob < self.logprob_threshold:
                logger.debug(
                    f"Filtered segment (avg_logprob={avg_logprob:.2f}): {text[:50]}"
                )
                continue

            # Check minimum duration
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)
            duration = end - start
            if duration < self.min_segment_duration:
                logger.debug(
                    f"Filtered segment (duration={duration:.2f}s): {text[:50]}"
                )
                continue

            # Check for consecutive exact repetitions
            if text == prev_text:
                repetition_count += 1
                if repetition_count >= self.max_repetitions:
                    logger.debug(
                        f"Filtered segment (repetition #{repetition_count}): {text[:50]}"
                    )
                    continue
            else:
                repetition_count = 0
                prev_text = text

            filtered.append(segment)

        original_count = len(segments)
        filtered_count = len(filtered)
        removed_count = original_count - filtered_count

        if removed_count > 0:
            logger.info(
                f"Filtered {removed_count} segments "
                f"({original_count} -> {filtered_count})"
            )

        return filtered

    def get_segments(self, audio_path: str) -> List[Dict[str, Any]]:
        result = self.transcribe(audio_path)
        segments = result.get("segments", [])
        return self.filter_segments(segments)
