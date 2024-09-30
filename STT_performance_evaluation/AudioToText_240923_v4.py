import speech_recognition as sr
from google.cloud import speech
from vosk import Model, KaldiRecognizer
import whisper
import wave
import json
import io
from tkinter import Tk, Button, Label, filedialog, messagebox
from pydub import AudioSegment
import os
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import numpy as np

def convert_m4a_to_wav(m4a_file_path):
    audio = AudioSegment.from_file(m4a_file_path, format="m4a")
    wav_file_path = m4a_file_path.replace(".m4a", ".wav")
    audio.export(wav_file_path, format="wav")
    return wav_file_path

def transcribe_audio_speech_recognition(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='ko-KR')
        return text

def transcribe_audio_sphinx(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_sphinx(audio_data, language='ko-KR')
            return text
        except sr.UnknownValueError:
            return "음성을 인식할 수 없습니다."
        except sr.RequestError as e:
            return f"Sphinx 오류; {e}"

def transcribe_audio_google_cloud(file_path):
    client = speech.SpeechClient()

    with io.open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="ko-KR",
    )

    response = client.recognize(config=config, audio=audio)
    return " ".join([result.alternatives[0].transcript for result in response.results])

def transcribe_audio_vosk(file_path):
    model = Model(model_path="C:\\vosk-model-small-ko-0.22")  # 한국어 모델 사용
    wf = wave.open(file_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            results.append(result['text'])
    
    final_result = json.loads(rec.FinalResult())
    results.append(final_result['text'])
    
    return " ".join(results)

def transcribe_audio_whisper(file_path):
    model = whisper.load_model("base")  # 또는 "small", "medium", "large" 모델 선택
    result = model.transcribe(file_path)
    return result["text"]

def transcribe_audio_wav2vec(file_path):
    # Load pre-trained model and processor
    processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
    model = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

    # Load audio
    audio, rate = librosa.load(file_path, sr=16000)

    # Tokenize
    input_values = processor(audio, return_tensors="pt", padding="longest").input_values

    # Retrieve logits
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

def process_file(file_path):
    try:
        wav_file_path = convert_m4a_to_wav(file_path)
        
        # Speech Recognition (Google) 라이브러리 사용
        sr_transcription = transcribe_audio_speech_recognition(wav_file_path)
        sr_output_file = wav_file_path.replace(".wav", "_sr_google_transcription.txt")
        with open(sr_output_file, "w", encoding='utf-8') as file:
            file.write("Speech Recognition (Google) 결과:\n" + sr_transcription)
        
        # Speech Recognition (Sphinx) 라이브러리 사용
        sphinx_transcription = transcribe_audio_sphinx(wav_file_path)
        sphinx_output_file = wav_file_path.replace(".wav", "_sr_sphinx_transcription.txt")
        with open(sphinx_output_file, "w", encoding='utf-8') as file:
            file.write("Speech Recognition (Sphinx) 결과:\n" + sphinx_transcription)
        
        # Google Cloud Speech-to-Text API 사용
        gc_transcription = transcribe_audio_google_cloud(wav_file_path)
        gc_output_file = wav_file_path.replace(".wav", "_gc_transcription.txt")
        with open(gc_output_file, "w", encoding='utf-8') as file:
            file.write("Google Cloud Speech-to-Text 결과:\n" + gc_transcription)
        
        # Vosk 사용
        vosk_transcription = transcribe_audio_vosk(wav_file_path)
        vosk_output_file = wav_file_path.replace(".wav", "_vosk_transcription.txt")
        with open(vosk_output_file, "w", encoding='utf-8') as file:
            file.write("Vosk 결과:\n" + vosk_transcription)

        # Whisper 사용
        whisper_transcription = transcribe_audio_whisper(wav_file_path)
        whisper_output_file = wav_file_path.replace(".wav", "_whisper_transcription.txt")
        with open(whisper_output_file, "w", encoding='utf-8') as file:
            file.write("Whisper 결과:\n" + whisper_transcription)

        # wav2vec 사용
        wav2vec_transcription = transcribe_audio_wav2vec(wav_file_path)
        wav2vec_output_file = wav_file_path.replace(".wav", "_wav2vec_transcription.txt")
        with open(wav2vec_output_file, "w", encoding='utf-8') as file:
            file.write("wav2vec 결과:\n" + wav2vec_transcription)
        
        os.remove(wav_file_path)
        return True
    except Exception as e:
        messagebox.showerror("오류", str(e))
        return False

def process_folder(folder_path):
    success_count = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".m4a"):
            file_path = os.path.join(folder_path, file_name)
            if process_file(file_path):
                success_count += 1
    messagebox.showinfo("완료", f"모든 변환이 완료되었습니다! {success_count}개의 파일이 성공적으로 처리되었습니다.")

def select_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        process_folder(folder_path)
    root.destroy()

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("M4A 파일", "*.m4a")])
    if file_path:
        if process_file(file_path):
            messagebox.showinfo("완료", "변환이 완료되었습니다! 텍스트가 저장되었습니다.")
    root.destroy()

root = Tk()
root.title("오디오 텍스트 변환기")

Label(root, text="진행할 옵션을 선택하세요:", padx=20, pady=20).pack()

Button(root, text="폴더 선택", command=select_folder, padx=10, pady=5).pack(pady=10)
Button(root, text="파일 선택", command=select_file, padx=10, pady=5).pack(pady=10)

root.mainloop()