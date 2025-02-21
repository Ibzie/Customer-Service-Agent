import whisper
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class ASRConfig:
    device: str = "cuda"
    language: Optional[str] = None

class ASRProcessor:
    def __init__(self, config: ASRConfig = ASRConfig()):
        self.config = config
        self.model = whisper.load_model("turbo")
    
    def transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        # Normalize audio
        audio = audio_chunk.astype(np.float32) / np.max(np.abs(audio_chunk))
        
        # Pad/trim to 30 seconds
        audio = whisper.pad_or_trim(audio)
        
        # Create mel spectrogram using model's dimensions
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels).to(self.model.device)
        
        # Decode audio
        options = whisper.DecodingOptions(language=self.config.language)
        result = whisper.decode(self.model, mel, options)
        
        return result.text.strip()