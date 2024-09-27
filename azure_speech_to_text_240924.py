import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment

# m4a 파일을 wav로 변환
sound = AudioSegment.from_file("통화 녹음 0537404675_231030_174803.m4a", format="m4a")
sound.export("converted_audio.wav", format="wav")

# Azure API 설정
speech_config = speechsdk.SpeechConfig(subscription="05bb53b4464c41cf90ab4330c2f598e5", region="koreacentral")
speech_config.speech_recognition_language = "ko-KR"  # 한국어로 설정

audio_config = speechsdk.AudioConfig(filename="converted_audio.wav")
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# 연속 인식을 위한 설정
all_results = []

def recognized_cb(evt):
    all_results.append(evt.result.text)
    print('RECOGNIZED: {}'.format(evt.result.text))

speech_recognizer.recognized.connect(recognized_cb)

# 연속 인식 시작
print('인식 시작...')
speech_recognizer.start_continuous_recognition()

import time
time.sleep(int(sound.duration_seconds) + 5)  # 오디오 길이 + 5초 대기

speech_recognizer.stop_continuous_recognition()

# 결과를 파일에 저장
azure_text_filename = 'azure_converted_audio.txt'
with open(azure_text_filename, 'w', encoding='utf-8') as f:
    for result in all_results:
        f.write(result + "\n")

print(f"텍스트가 '{azure_text_filename}'에 저장되었습니다.")