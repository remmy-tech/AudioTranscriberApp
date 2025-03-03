import ssl
import certifi
import os
import tempfile
from datetime import datetime
import streamlit as st
import whisper

# Set up SSL context using certifi's certificates.
def get_ssl_context(*args, **kwargs):
    return ssl.create_default_context(cafile=certifi.where())

ssl._create_default_https_context = get_ssl_context

# Cache the model loading so that the model is only loaded once per session.
@st.cache_resource
def load_whisper_model(model_name: str):
    return whisper.load_model(model_name)

st.title("Whisper Audio Transcription App")
st.write("Select a Whisper model, upload your audio files, and download your transcripts. Simple as that!")

# Allow the user to choose a Whisper model.
model_options = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large"]
selected_model = st.selectbox("Select a Whisper Model", model_options)

# Load the selected model.
model = load_whisper_model(selected_model)

# File uploader for multiple audio files.
uploaded_files = st.file_uploader("Upload Audio Files", accept_multiple_files=True, type=["mp3", "wav", "m4a", "flac", "ogg"])

if uploaded_files:
    st.write("### Uploaded Files")
    for file in uploaded_files:
        st.write(f"- {file.name}")

def process_transcription(file, model, idx):
    st.write(f"**Processing file: {file.name}**")
    # Save the uploaded file to a temporary file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
        temp_file.write(file.getbuffer())
        temp_file_path = temp_file.name

    try:
        # Load the audio to verify that it's valid.
        audio = whisper.load_audio(temp_file_path)
        if audio.size == 0:
            raise ValueError("The audio file is empty or could not be processed. "
                             "Please verify the file or ensure FFmpeg is installed.")

        with st.spinner(f"Transcribing {file.name}..."):
            # Transcribe the already-loaded audio array.
            result = model.transcribe(audio)
            transcript = result.get("text", "No transcript generated.")
    except Exception as e:
        transcript = f"Error transcribing file: {e}"
    finally:
        os.remove(temp_file_path)
    
    return transcript

# Button to start transcription.
if uploaded_files and st.button("Transcribe Files"):
    for idx, uploaded_file in enumerate(uploaded_files):
        transcript = process_transcription(uploaded_file, model, idx)
        
        # Display the transcript in an expandable section.
        with st.expander(f"Transcript for {uploaded_file.name}"):
            st.text_area("Transcript", transcript, height=200, key=f"transcript_{idx}")

        # Create a unique filename for the transcript.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_filename = f"{os.path.splitext(uploaded_file.name)[0]}_transcript_{timestamp}.txt"
        
        # Provide a download button.
        st.download_button(
            label="Download Transcript",
            data=transcript,
            file_name=transcript_filename,
            mime="text/plain",
            key=f"download_{idx}"
        )