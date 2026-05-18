import whisper
from typing import List, Dict, Optional, Any, Set
from collections import Counter
from difflib import SequenceMatcher
import logging
import tempfile
import wave
from pathlib import Path

logger = logging.getLogger(__name__)


# Default thresholds for filtering hallucinated/low-quality segments
# These are tuned to catch obvious hallucinations while preserving quiet speech
DEFAULT_NO_SPEECH_THRESHOLD = 0.9  # Filter segments with no_speech_prob > this
DEFAULT_LOGPROB_THRESHOLD = -1.5  # Filter segments with avg_logprob < this
DEFAULT_TEMPERATURE = 0.0  # Deterministic decoding reduces hallucinations
DEFAULT_COMPRESSION_RATIO_THRESHOLD = 2.4  # Filter repetitive/degenerate output
DEFAULT_MIN_DURATION = 0.1  # Minimum segment duration in seconds
DEFAULT_MAX_REPETITIONS = 2  # Max times same text can repeat consecutively
DEFAULT_SIMILARITY_THRESHOLD = (
    0.7  # Text similarity threshold for detecting similar hallucinations
)
DEFAULT_REPETITION_WINDOW = 60.0  # Time window (seconds) to look for similar text
ADJACENT_DUPLICATE_GAP = 1.0  # Seconds between adjacent near-duplicates
ADJACENT_DUPLICATE_LENGTH = 40  # Min text length for adjacent duplicate filter
ADJACENT_DUPLICATE_SIMILARITY = 0.92  # Similarity for adjacent duplicate filter
ADJACENT_DUPLICATE_DURATION = 5.0  # Long repeated segments are more suspect
REPEATED_TOKEN_MIN_COUNT = 8  # Min token count for repeated-token hallucination
REPEATED_TOKEN_RATIO = 0.8  # Dominant token ratio for repeated-token filter
REPEATED_CHAR_MIN_LENGTH = 12  # Min text length for repeated-char filter
REPEATED_CHAR_RATIO = 0.85  # Dominant character ratio for repeated-char filter
SUSPICIOUS_COMPRESSION_RATIO = 4.0  # High compression often signals hallucination
SUSPICIOUS_NO_SPEECH_PROB = 0.6  # Suspicious silence score for clip validation
SUSPICIOUS_LOGPROB = -1.3  # Suspicious confidence score for clip validation
SUSPICIOUS_SHORT_WORDS = 3  # Short fragments need extra scrutiny
SUSPICIOUS_MIN_DURATION = 1.0  # Ignore micro-fragments in validation pass
CLIP_VALIDATION_PADDING = 0.35  # Context padding around suspicious segments
CLIP_VALIDATION_SIMILARITY = 0.82  # Min similarity to keep suspicious segment
SHORT_PHRASE_REPEAT_COUNT = 4  # Repeated short phrases are often hallucinations


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
        temperature: float = DEFAULT_TEMPERATURE,
        compression_ratio_threshold: float = DEFAULT_COMPRESSION_RATIO_THRESHOLD,
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
        self.temperature = temperature
        self.compression_ratio_threshold = compression_ratio_threshold
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
        # Lower temperature improves determinism for subtitle generation
        # compression_ratio_threshold filters out repetitive segments
        # condition_on_previous_text=False prevents context-based repetitions
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=True,
            temperature=self.temperature,
            compression_ratio_threshold=self.compression_ratio_threshold,
            logprob_threshold=self.logprob_threshold,
            no_speech_threshold=self.no_speech_threshold,
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

    @staticmethod
    def has_repeated_token_hallucination(text: str) -> bool:
        """Detect segments dominated by one short token repeated many times."""
        tokens = [token.strip(".,!?;:-").lower() for token in text.split()]
        tokens = [token for token in tokens if token]
        if len(tokens) < REPEATED_TOKEN_MIN_COUNT:
            return False

        most_common_token, count = Counter(tokens).most_common(1)[0]
        if len(most_common_token) > 5:
            return False

        return (count / len(tokens)) >= REPEATED_TOKEN_RATIO

    @staticmethod
    def has_repeated_char_hallucination(text: str) -> bool:
        """Detect segments dominated by one repeated character like HMMMMM."""
        compact = "".join(ch.lower() for ch in text if ch.isalpha())
        if len(compact) < REPEATED_CHAR_MIN_LENGTH:
            return False

        most_common_char, count = Counter(compact).most_common(1)[0]
        if most_common_char not in {"h", "m", "a", "o"}:
            return False

        return (count / len(compact)) >= REPEATED_CHAR_RATIO

    @staticmethod
    def is_suspicious_segment(text: str, segment: Dict[str, Any]) -> bool:
        """Identify segments worth validating in isolation."""
        words = text.split()
        duration = segment.get("end", 0.0) - segment.get("start", 0.0)
        no_speech_prob = segment.get("no_speech_prob", 0.0)
        avg_logprob = segment.get("avg_logprob", 0.0)
        compression_ratio = segment.get("compression_ratio", 0.0)

        if (
            compression_ratio >= SUSPICIOUS_COMPRESSION_RATIO
            and (
                len(words) >= 4
                or no_speech_prob >= 0.2
                or avg_logprob <= -0.8
            )
        ):
            return True

        if (
            len(words) <= SUSPICIOUS_SHORT_WORDS
            and duration >= SUSPICIOUS_MIN_DURATION
            and (
                no_speech_prob >= SUSPICIOUS_NO_SPEECH_PROB
                or avg_logprob <= SUSPICIOUS_LOGPROB
            )
        ):
            return True

        if (
            duration >= SUSPICIOUS_MIN_DURATION
            and no_speech_prob >= 0.75
            and avg_logprob <= -0.8
        ):
            return True

        return False

    def _slice_audio_clip(self, audio_path: str, start: float, end: float) -> str:
        """Write a temporary WAV clip for segment re-validation."""
        with wave.open(audio_path, "rb") as input_wav:
            frame_rate = input_wav.getframerate()
            channels = input_wav.getnchannels()
            sample_width = input_wav.getsampwidth()
            start_frame = max(0, int(start * frame_rate))
            end_frame = int(end * frame_rate)
            input_wav.setpos(start_frame)
            frames = input_wav.readframes(max(0, end_frame - start_frame))

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()
        with wave.open(temp_file.name, "wb") as output_wav:
            output_wav.setnchannels(channels)
            output_wav.setsampwidth(sample_width)
            output_wav.setframerate(frame_rate)
            output_wav.writeframes(frames)

        return temp_file.name

    def _transcribe_clip(self, clip_path: str) -> List[str]:
        """Transcribe a short validation clip with permissive thresholds."""
        result = self.model.transcribe(
            clip_path,
            language=self.language,
            word_timestamps=True,
            temperature=self.temperature,
            compression_ratio_threshold=self.compression_ratio_threshold,
            logprob_threshold=-2.0,
            no_speech_threshold=0.95,
            condition_on_previous_text=False,
        )
        return [
            segment.get("text", "").strip()
            for segment in result.get("segments", [])
            if segment.get("text", "").strip()
        ]

    def _passes_clip_validation(self, audio_path: str, text: str, segment: Dict[str, Any]) -> bool:
        """Re-transcribe a suspicious segment in isolation and compare text."""
        start = max(0.0, segment.get("start", 0.0) - CLIP_VALIDATION_PADDING)
        end = segment.get("end", 0.0) + CLIP_VALIDATION_PADDING
        clip_path = self._slice_audio_clip(audio_path, start, end)
        try:
            clip_texts = self._transcribe_clip(clip_path)
        finally:
            Path(clip_path).unlink(missing_ok=True)

        if not clip_texts:
            return False

        best_similarity = max(
            self.text_similarity(text, clip_text) for clip_text in clip_texts
        )
        return best_similarity >= CLIP_VALIDATION_SIMILARITY

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

    def filter_segments(
        self, segments: List[Dict[str, Any]], audio_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
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
        short_phrase_counts = Counter(
            segment.get("text", "").strip().lower()
            for segment in segments
            if segment.get("text", "").strip()
        )
        filtered = []
        prev_text = None
        repetition_count = 0

        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            if not text:
                continue
            lower_text = text.lower()

            if self.has_repeated_token_hallucination(text):
                logger.debug(
                    f"Filtered segment (repeated-token hallucination): {text[:50]}"
                )
                continue

            if self.has_repeated_char_hallucination(text):
                logger.debug(
                    f"Filtered segment (repeated-char hallucination): {text[:50]}"
                )
                continue

            if (
                len(text.split()) <= SUSPICIOUS_SHORT_WORDS
                and short_phrase_counts[lower_text] >= SHORT_PHRASE_REPEAT_COUNT
                and (
                    len(text.split()) == 1
                    or segment.get("avg_logprob", 0.0) <= -1.2
                    or segment.get("no_speech_prob", 0.0) >= 0.5
                )
            ):
                logger.debug(
                    f"Filtered segment (repeated short phrase): {text[:50]}"
                )
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

            if filtered:
                prev_segment = filtered[-1]
                prev_filtered_text = prev_segment.get("text", "").strip()
                gap = start - prev_segment.get("end", 0.0)
                prev_duration = prev_segment.get("end", 0.0) - prev_segment.get(
                    "start", 0.0
                )
                if (
                    gap <= ADJACENT_DUPLICATE_GAP
                    and len(prev_filtered_text) >= ADJACENT_DUPLICATE_LENGTH
                    and len(text) >= ADJACENT_DUPLICATE_LENGTH
                    and (
                        prev_duration >= ADJACENT_DUPLICATE_DURATION
                        or duration >= ADJACENT_DUPLICATE_DURATION
                    )
                ):
                    similarity = self.text_similarity(prev_filtered_text, text)
                    if similarity >= ADJACENT_DUPLICATE_SIMILARITY:
                        logger.debug(
                            f"Filtered segment (adjacent duplicate): {text[:50]}"
                    )
                        continue

            if (
                audio_path is not None
                and self.is_suspicious_segment(text, segment)
                and not self._passes_clip_validation(audio_path, text, segment)
            ):
                logger.debug(
                    f"Filtered segment (clip validation failed): {text[:50]}"
                )
                continue

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
        return self.filter_segments(segments, audio_path=audio_path)
