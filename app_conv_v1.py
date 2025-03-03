import ssl
import certifi

def get_ssl_context(*args, **kwargs):
    return ssl.create_default_context(cafile=certifi.where())

ssl._create_default_https_context = get_ssl_context

import os
import tempfile
from datetime import datetime
import streamlit as st
import whisper

# Cache the model loading to speed up subsequent loads.
@st.cache_resource
def load_whisper_model(model_name: str):
    return whisper.load_model(model_name)

# App title and description.
st.title("Whisper Audio Transcription App")
st.write("Upload your audio files, select a Whisper model, and download your transcripts.")

# Let the user choose which Whisper model to use.
model_options = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large"]
selected_model = st.selectbox("Select a Whisper Model", model_options)

# Load the selected model.
model = load_whisper_model(selected_model)

# File uploader to allow multiple audio files.
uploaded_files = st.file_uploader("Upload Audio Files", accept_multiple_files=True, type=["mp3", "wav", "m4a", "flac", "ogg"])

# Display the list of uploaded files.
if uploaded_files:
    st.write("### Uploaded Files")
    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")

# Button to start transcription.
if uploaded_files and st.button("Transcribe Files"):
    for uploaded_file in uploaded_files:
        st.write(f"**Processing file: {uploaded_file.name}**")
        # Save the uploaded file temporarily.
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name

        # Transcribe the audio file.
        try:
            result = model.transcribe(temp_file_path)
            transcript = result["text"]
        except Exception as e:
            transcript = f"Error transcribing file: {e}"
        
        # Display the transcript.
        st.text_area("Transcript", transcript, height=200, key=uploaded_file.name)
        
        # Create a unique filename for the transcript.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_filename = f"{os.path.splitext(uploaded_file.name)[0]}_transcript_{timestamp}.txt"
        
        # Provide a download button for the transcript.
        st.download_button(
            label="Download Transcript",
            data=transcript,
            file_name=transcript_filename,
            mime="text/plain",
            key=uploaded_file.name + "_download"
        )