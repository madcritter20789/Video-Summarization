import whisper
import os
'''
def transcribe_audio(audio_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    with open(output_path, "w") as f:
        f.write(result['text'])
'''
def transcribe_audio(audio_path, transcript_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(result['text'])
    print(f"Transcription saved to {transcript_path}")

# Example usage
# transcribe_audio('data/audio/sample.mp3', 'data/transcripts/sample.txt')
