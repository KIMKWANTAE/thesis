import os
import hashlib
import tkinter as tk
from tkinter import filedialog, messagebox
import csv

def calculate_md5(file_path):
    """파일의 MD5 해시를 계산합니다."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def get_md5_for_folder(folder_path):
    """폴더 내 모든 파일의 MD5 해시를 계산합니다."""
    file_hashes = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_hash = calculate_md5(file_path)
                file_hashes[file_path] = file_hash
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return file_hashes

def select_folder():
    """폴더 선택 대화상자를 열고 선택된 폴더의 MD5 해시를 계산합니다."""
    folder_path = filedialog.askdirectory()
    if folder_path:
        results = get_md5_for_folder(folder_path)
        save_results_to_csv(results, folder_path)
        display_results(results)
    else:
        messagebox.showinfo("알림", "폴더가 선택되지 않았습니다.")

def save_results_to_csv(results, folder_path):
    """결과를 CSV 파일로 저장합니다."""
    csv_file_path = os.path.join(folder_path, "md5_hashes.csv")
    with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["순번", "파일명", "MD5 해시값"])
        for index, (file_path, md5_hash) in enumerate(results.items(), start=1):
            file_name = os.path.basename(file_path)
            csv_writer.writerow([index, file_name, md5_hash])
    messagebox.showinfo("알림", f"결과가 {csv_file_path}에 저장되었습니다.")

def display_results(results):
    """결과를 새 창에 표시합니다."""
    result_window = tk.Toplevel(root)
    result_window.title("MD5 해시 결과")
    result_window.geometry("800x600")
    
    text_widget = tk.Text(result_window, wrap=tk.WORD, font=("Arial", 12))
    text_widget.pack(expand=True, fill='both')
    
    scrollbar = tk.Scrollbar(result_window, command=text_widget.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_widget.config(yscrollcommand=scrollbar.set)
    
    text_widget.insert(tk.END, "순번\t파일명\tMD5 해시값\n")
    text_widget.insert(tk.END, "-" * 80 + "\n")
    
    for index, (file_path, md5_hash) in enumerate(results.items(), start=1):
        file_name = os.path.basename(file_path)
        text_widget.insert(tk.END, f"{index}\t{file_name}\t{md5_hash}\n")
    
    text_widget.config(state=tk.DISABLED)  # 읽기 전용으로 설정

# 메인 윈도우 생성
root = tk.Tk()
root.title("폴더 MD5 해시 계산기")
root.geometry("500x300")  # 창 크기를 500x300으로 유지

# 프레임을 생성하여 버튼을 중앙에 배치
frame = tk.Frame(root)
frame.place(relx=0.5, rely=0.5, anchor='center')

select_button = tk.Button(frame, text="폴더 선택", command=select_folder, 
                          font=("Arial", 16), padx=20, pady=10)
select_button.pack()

root.mainloop()