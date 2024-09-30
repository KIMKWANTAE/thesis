import speech_recognition as sr
from tkinter import Tk, Button, Label, filedialog, messagebox
from pydub import AudioSegment
import os

def convert_m4a_to_wav(m4a_file_path):
    audio = AudioSegment.from_file(m4a_file_path, format="m4a")
    wav_file_path = m4a_file_path.replace(".m4a", ".wav")
    audio.export(wav_file_path, format="wav")
    return wav_file_path

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='ko-KR')
        return text

def process_file(file_path):
    try:
        wav_file_path = convert_m4a_to_wav(file_path)
        transcription = transcribe_audio(wav_file_path)
        output_file = wav_file_path.replace(".wav", "_transcription.txt")
        with open(output_file, "w", encoding='utf-8') as file:
            file.write(transcription)
        os.remove(wav_file_path)
        return True  # 성공을 나타냄
    except Exception as e:
        messagebox.showerror("오류", str(e))
        return False  # 실패를 나타냄

def process_folder(folder_path):
    success_count = 0  # 성공적으로 처리된 파일 수를 계산
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".m4a"):
            file_path = os.path.join(folder_path, file_name)
            if process_file(file_path):
                success_count += 1
    # 모든 파일 처리가 완료된 후 알림을 표시
    messagebox.showinfo("완료", f"모든 변환이 완료되었습니다! {success_count}개의 파일이 성공적으로 처리되었습니다.")

def select_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        process_folder(folder_path)
    root.destroy()

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("M4A 파일", "*.m4a")])
    if file_path:
        process_file(file_path)
        messagebox.showinfo("완료", "변환이 완료되었습니다! 텍스트가 저장되었습니다.")
    root.destroy()

root = Tk()
root.title("오디오 텍스트 변환기")

Label(root, text="진행할 옵션을 선택하세요:", padx=20, pady=20).pack()

Button(root, text="폴더 선택", command=select_folder, padx=10, pady=5).pack(pady=10)
Button(root, text="파일 선택", command=select_file, padx=10, pady=5).pack(pady=10)

root.mainloop()
