import streamlit as st
import whisper
import tempfile
import os

def main():
    st.title("Whisper Audio Transcription App")
    st.write("Transcribe your audio files using OpenAI's Whisper model.")

    # Allow user to select a model from the available options
    model_options = ["tiny", "base", "small", "medium", "large"]
    model_choice = st.selectbox("Select the Whisper model", options=model_options)

    # Upload an audio file
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "mp4", "ogg"])
    
    if uploaded_file is not None:
        # Display the uploaded audio
        st.audio(uploaded_file)
        
        if st.button("Transcribe"):
            with st.spinner("Transcribing..."):
                # Determine file extension for the temporary file
                suffix = os.path.splitext(uploaded_file.name)[1] if uploaded_file.name else ".wav"
                # Save the uploaded file to a temporary file so Whisper can process it
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Load the selected Whisper model (this may take some time)
                model = whisper.load_model(model_choice)
                # Transcribe the audio file
                result = model.transcribe(tmp_path)
                transcript = result["text"]

                # Cleanup temporary file
                os.remove(tmp_path)

            st.success("Transcription complete!")
            # Display transcript in a text area
            st.text_area("Transcript", transcript, height=300)
            # Provide a download button for the transcript
            st.download_button("Download Transcript", transcript, file_name="transcript.txt")

if __name__ == "__main__":
    main()