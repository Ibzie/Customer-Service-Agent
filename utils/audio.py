# utils/audio.py
import numpy as np
from scipy import signal
import io
import wave

def process_uploaded_audio(uploaded_file) -> np.ndarray:
    # Read bytes from UploadedFile
    audio_bytes = uploaded_file.read()
    
    # Convert to wave format
    with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        audio_data = wav_file.readframes(wav_file.getnframes())
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
    # Convert to float32 and normalize
    audio_array = audio_array.astype(np.float32) / 32768.0
    
    # Resample if needed
    if sample_rate != 16000:
        audio_array = signal.resample(audio_array, 
                                    int(len(audio_array) * 16000 / sample_rate))
    
    return audio_array

def detect_speech(audio: np.ndarray, frame_size: int = 400) -> np.ndarray:
    # Simple energy-based VAD
    energy = np.array([
        np.sum(audio[i:i+frame_size]**2) 
        for i in range(0, len(audio), frame_size)
    ])
    
    threshold = np.mean(energy) * 0.5
    is_speech = energy > threshold
    
    # Get speech segments
    speech_audio = np.concatenate([
        audio[i*frame_size:(i+1)*frame_size]
        for i, speech in enumerate(is_speech) if speech
    ])
    
    return speech_audio