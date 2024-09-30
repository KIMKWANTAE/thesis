import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# Defining the true reference transcription from the answer file
reference_transcription = """네. 김관태입니다. 네. 안녕하십니까. 여기 마약 수사관 김홍수 수사관입니다. 아 네네. 혹시 지금 전화 통화 가능하신가요? 아 네네. 다름이 아니라 내일 오전에 그 모바일 선별 작업하려고 전화드렸거든요. 아 네. 혹시 내일 오전에 시간 가능하실까요? 아 네네. 제가 있어요. 아네. 알겠습니다. 그거 제가 지금 조퇴해가지고 그 쪽지로 누구 거 모델 사용자 명이랑 모델명 남겨놓으면 내가 준비해 놓을게요. 혹시 몇 시쯤에 오실 거예요? 내일 한 9시 반에서 10시 사이에 가려고. 아 네네. 그거 미리 쪽지로 보내놓으면 내가 미리 켜놓을게요. 아네. 알겠습니다. 아네. 네. 수고하십시오. 네."""

# Transcriptions from various STT models
transcriptions = {
    'Azure': """네 김하태입니다. 안녕하십니까? 여기 마약수사 김홍수입니다. 아 네네 아, 혹시 지금 전화 통화 가능하신가요? 아 네네 아 다름이 아니라 내일 오전에 그 모바일 선별 작업하려고 전화드렸거든요. 아 네, 그 혹시 내일 오전에 시간 가능하실까요? 아 네네, 제가 있어요. 아 네, 알겠습니다. 제가. 지금 조퇴해가지고 그 쪽지로 누구 거 모델 이렇게 사용자 명이랑 모델명 남겨놓으면 그 내가 준비해 놓을게요. 혹시 몇 시쯤에 오실 거예요?한 아홉 시 반에서 열 시 사이에 가려고요. 아 네네 그거 미리 쪽지랑 쪽지로 보내놓으면 내가 미리 켜놓을게요. 아 네, 알겠습니다. 아 네 수고하십시오. 네.""",
    'Google Cloud': """내 곁에입니다입니다 아네 혹시 지금 전화 통화 가능하신가요 내일 오전에 그 모바일 선별 작업 하려고 전화 드렸거든요 내일 오전에 시간 가능하실까요 제가 있어요네 알겠습니다 제가 지금 해 가지고 그 쪽지로 누구 거 모델 이렇게 사용자 명이랑 모델명 남겨 놓으면 내가 준비해 놓을게요 혹시 몇 시쯤에 오실 거예요 내일 9시 반에서 1시 사이에 가려고 해 놓으면 놓을게요 알겠습니다네""",
    'SpeechRecognition (Google)': """네 괜찮습니다 마약 수정 수사관입니다 아네네 혹시 지금 전화 통화 가능하신가요 아네네 다름이 아니라 내일 오전에 그 모바일 선별 작업하려고 전화 드렸거든요 내일 오전에 시간 가능하실까요 아네네 제가 있어요 아네 알겠습니다 제가 지금 해가지고 그 쪽지로 누구 거 모델 사용자명이 모델명 남겨 놓으면 내가 준비해 놓을게요 혹시 몇 시쯤에 오실 거예요 9시 반에서 10시 사이에 가려고 해 놓으면 내가 미리 켜 놓을게요 아네 알겠습니다 아네 수고하십시오""",
    'wav2vec': """네 팀전냐  마약스한색 기능수사닙니아시 쯤 단 통화가는하신가아다르면 네일 오전네 그 모바일 손별차가 관료  전 노력화습아네네 오전이 시가능화실까아베 제가 있어아 알혔습니후 어진 손 내가지고 그 욕시로 누국과 모델 사용장 명이라고해 명란 교도 우와  내 충계에 놓 깨 호시 을 시쯤 에고 실까요아홉 시 판에서일 시 사과네꾸고 미리 토시를 학쪽 교건에 미와 레우비리 여홉 개 알렸습니래   네 요""",
    'Vosk': """엘비라 아령 교원 말도안 돼 공포 한 입니다 아아 뢰베 가 는데 뒷굽 카나타 가 나 인간 아벨 의 아르메니아 를 오도넬 그 모바일 선별 적 빨려 을 달아 달아 크림을 아래 넬 오던 액션 가능한 할까요 아베베 혜란 안에 하였습니다 고고 어림 네가지 과거 역갤러 꺾어 머렐 하울 양 이란걸 액을 남겨둬 야 라 쪽에 더 개혁 이란 게 바로 일까요 한편 파업위기 B 반 돌리 하얀 가루가 백여 고고 어빌 쪼개 적격 거래 했어야 여기 를 여러 개 하나요 스미다 아래 의 골격 에 의 해 """,
    'Whisper': """네. 김여태입니다. 네. 안녕하십니까. 여기 마약 수사원 김홍수 수사원입니다. 네. 혹시 지금 전화 통화 가능하신가요? 네. 다름이 아니라 내일 오전에 모바일 선별 작업하려고 전화드렸거든요. 네. 혹시 내일 오전에 시간 가능하실까요? 네. 제가 있어요. 네. 알겠습니다. 그거 제가 지금 조퇴해가지고 그 쪽지로 누구 거 모델 사용자 명이랑 모델명 남겨놓으면 내가 준비해 놓을게요. 혹시 몇 시쯤에 오실 거예요? 내일 한 9시 반에서 10시 사이에 가려고. 네. 그거 미리 쪽지로 보내놓으면 내가 미리 켜놓을게요. 네. 알겠습니다. 네. 네. 수고하십시오. 네.""",
    'Clova Note': """네 김현태입니다. 네 안녕하십니까? 여기 마약 수사관실 김홍수 수사관입니다. 네네 혹시 지금 전화 통화 가능하신가요? 네네 다름이 아니라 내일 오전에 모바일 선별 작업하려고 전화드렸거든요. 아 네. 혹시 내일 오전에 시간 가능하실까요? 네네 제가 있어요. 네 알겠습니다. 그거 제가 지금 좋대 가지고 쪽지로 누구 거 모델 이렇게 사용자 명이랑 모델명 남겨놓으면 내가 준비해 놓을게요. 시 몇 시쯤에 오실 거예요? 내일 한 9시 반에서 10시 사이에 가려고 네네. 그거 미리 지랑 쪽지로 보내놓으면 내가 미리 켜놓을게요. 네 알겠습니다. 네 네. 수고하십시오. 네."""
}

# Calculating precision, recall, and F1-score using corrected word-based comparison
def calculate_metrics_corrected(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # For precision: How many words in the hypothesis are correct (present in the reference)
    true_positives = len([word for word in hyp_words if word in ref_words])
    false_positives = len(hyp_words) - true_positives

    # For recall: How many reference words are correctly captured by the hypothesis
    false_negatives = len(ref_words) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# Recalculate the metrics for each transcription
performance_data_corrected = {lib: calculate_metrics_corrected(reference_transcription, transcription) for lib, transcription in transcriptions.items()}

# Prepare the corrected dataframe for display
performance_df_corrected = pd.DataFrame(performance_data_corrected, index=['Precision', 'Recall', 'F1-Score']).T

# Save the dataframe to a CSV file
performance_df_corrected.to_csv(r'D:\Python test\audio_evaluation\stt_performance_metrics.csv', encoding='utf-8-sig')

print("STT Model Performance Metrics (Corrected):")
print(performance_df_corrected)
