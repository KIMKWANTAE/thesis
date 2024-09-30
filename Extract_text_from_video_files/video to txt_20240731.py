import ffmpeg
import whisper
from tkinter import Tk, filedialog
import os

def extract_audio_from_video(video_path, audio_path):
    """
    Extract audio from video using ffmpeg.
    """
    ffmpeg.input(video_path).output(audio_path).run()

def transcribe_audio(audio_path):
    """
    Transcribe audio using OpenAI's Whisper model.
    """
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def format_transcription(transcription):
    """
    Format the transcription by adding a line break at the end of each sentence.
    """
    sentences = transcription.split('. ')
    formatted_transcription = '.\n'.join(sentences)
    return formatted_transcription

def extract_text_from_video(video_path):
    """
    Complete process to extract text from video.
    """
    audio_path = "extracted_audio.mp3"
    extract_audio_from_video(video_path, audio_path)
    transcription = transcribe_audio(audio_path)
    formatted_transcription = format_transcription(transcription)
    return formatted_transcription

def save_transcription_to_file(transcription, video_path):
    """
    Save the transcription to a .txt file with the same name as the video file.
    """
    base_name = os.path.splitext(video_path)[0]
    txt_file_path = f"{base_name}.txt"
    with open(txt_file_path, 'w', encoding='utf-8') as f:
        f.write(transcription)
    print(f"Transcription saved to {txt_file_path}")

def select_file():
    """
    Open a file dialog to select an MP4 file.
    """
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select MP4 file",
        filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
    )
    return file_path

if __name__ == "__main__":
    video_path = select_file()
    if video_path:
        transcription = extract_text_from_video(video_path)
        print("Transcription:")
        print(transcription)
        save_transcription_to_file(transcription, video_path)
    else:
        print("No file selected.")
