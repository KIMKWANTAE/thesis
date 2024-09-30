import os
import cv2
import pytesseract
import numpy as np
from shutil import copy2
from tkinter import Tk, filedialog, messagebox, Label, Button
from PIL import Image, ImageEnhance

# Tesseract 경로 설정 (Windows의 경우 필요)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def select_folder(title):
    folder = filedialog.askdirectory(title=title)
    if not folder:
        messagebox.showerror("Error", f"No {title.lower()} selected. Exiting...")
        exit()
    return folder

def update_status(label, text):
    label.config(text=text)
    label.update_idletasks()

# Tkinter 초기화 및 숨김
root = Tk()
root.title("Image Text Classifier_By KKT")
root.geometry("400x200")

# 폴더 선택
input_folder = select_folder("Select Input Folder")
output_folder = select_folder("Select Output Folder")

# 상태 표시 레이블
status_label = Label(root, text="Processing...", font=('Helvetica', 12))
status_label.pack(pady=20)

# 완료 버튼 (초기에는 비활성화)
complete_button = Button(root, text="Open Output Folder", state="disabled", command=lambda: os.startfile(output_folder))
complete_button.pack(pady=20)

# 출력 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 이미지 파일 확장자 목록
image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

# 텍스트 파일 생성 여부 묻기
create_text_files = messagebox.askyesno("Create Text Files", "Do you want to create text files for each image with detected text?")

# 입력 폴더 내의 모든 파일 처리
for idx, filename in enumerate(os.listdir(input_folder)):
    # 파일 확장자 확인
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        # 이미지 파일 경로
        file_path = os.path.join(input_folder, filename)
        
        # 진행 상황 업데이트
        update_status(status_label, f"Processing {filename} ({idx + 1}/{len(os.listdir(input_folder))})")
        
        # PIL을 사용하여 이미지를 읽어들임
        try:
            pil_image = Image.open(file_path)
            
            # 이미지 전처리
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(2)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {file_path}\n{str(e)}")
            continue
        
        # 이미지가 제대로 읽어들여졌는지 확인
        if image is None:
            messagebox.showerror("Error", f"Error loading image: {file_path}")
            continue
        
        # 이미지 전처리: 그레이스케일 변환 및 이진화
        try:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image for OCR: {file_path}\n{str(e)}")
            continue
        
        # OCR을 사용하여 이미지에서 텍스트 추출
        try:
            # Tesseract 언어를 한국어와 영어로 설정
            text = pytesseract.image_to_string(binary_image, lang='kor+eng')
            # 텍스트가 존재하면 출력 폴더로 복사
            if text.strip():
                copy2(file_path, os.path.join(output_folder, filename))
                if create_text_files:
                    text_file_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.txt')
                    with open(text_file_path, 'w', encoding='utf-8') as text_file:
                        text_file.write(text)
        except Exception as e:
            messagebox.showerror("Error", f"Error extracting text from image {file_path}: {e}")

# 완료 메시지 및 버튼 활성화
update_status(status_label, "Processing complete!")
complete_button.config(state="normal")

messagebox.showinfo("Completed", "텍스트가 포함된 이미지 분류 및 텍스트 파일 생성 완료")

root.mainloop()
