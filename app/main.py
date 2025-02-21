# app/main.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from core.asr import ASRProcessor, ASRConfig
from utils.audio import process_uploaded_audio, detect_speech

def initialize_asr():
    if 'asr' not in st.session_state:
        st.session_state.asr = ASRProcessor(ASRConfig())

def main():
    st.title("Customer Service Agent - ASR Test")
    initialize_asr()

    audio_value = st.audio_input("Record your message")
    
    if audio_value:
        st.audio(audio_value)
        
        with st.spinner("Processing audio..."):
            audio_array = process_uploaded_audio(audio_value)
            transcript = st.session_state.asr.transcribe_chunk(audio_array)
            st.write("Transcription:", transcript)

if __name__ == "__main__":
    main()