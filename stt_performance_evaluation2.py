import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# Defining the true reference transcription from the answer file
reference_transcription = """김관태입니다 김계장 낸데 어 출장 갔네 통화 가능한가 10월 11일 날 수요일 날 내 당직인데 내가 그날 김천에 감사를 가는데 다른 사람한테 뭐 일단 이야기할 사람도 없고 해가지고 김 계장한테 먼저 얘기하는데 김 계장이 10월 24일날 화요일날 당직이던데 혹시 바꿀 수 있나 싶어서 10월 11일요 다음 주 수요일 날이 내 당직이거든 일정표 보고 바로 다시 전화드릴게요 오케이"""

# Transcriptions from various STT models
transcriptions = {
    'Clova note': """김하니 회장 데 출장 갔네 통화 가능한가 네 10월 11일 날 수요일 날 내 당직인데 내가 그날 김천에 감사를 가는데 다른 사람한테 뭐 일단 이야기할 사람도 없고 해가지고 김 계장한테 먼저 얘기하는데 김 계장이 10월 24일날 화요일날 당직이던데 혹시 바꿀 수 있나 싶어서 10월 11일요 다음 주 수요일 날이 내 당직이거든. 일정표 보고 바로 다시 전화드릴게요. 오케이.""",
    'Google Cloud': """제가 출장 갔네 10월 10일 수요일 날 내일 당직인데 그날 김천에 감사를 가는데 이야기 할 사람도 없고 해 가지고 경민이한테 문자 얘기하는데 김 회장이 12월 24일 날 화요일 날 장식 있던데 혹시 있나 싶어서 10월 11일 다음 주 수요일 날이 보고 바로 전화 드릴게요 그때""",
    'SpeechRecognition (Google)': """출장 갔네 통화 가능한가 10월 11일 날 수요일 날 당직 그날 김천에 감사를 가는데 다른 사람한테 뭐 일단 이야기할 사람도 없고 해 가지고 김경희한테 전화 드릴게요""",
    'Whisper Colab Large': """출장 갔네 통화 가능한가 10월 11일 수요일 내 당직인데 김천에 감사를 가는데 다른 사람한테 뭐 일단 이야기할 사람도 없고 해가지고 김계장한테 먼저 얘기하는데 김계장이 10월 24일날 화요일날 당직이던데 혹시 바꿀 수 있나 싶어서 다음 주 수요일 날이 내 당직이거든""",
    'Whisper': """그냥 출장 갔네 더 하는 한강 저 10월 11일 나 10월 날 당직인데 그러니까 그날 김천에 감사를 가는데 다른 사람한테 뭘 일단 이야기하는 사람도 없고 그래서 김현한한테 뭔지 얘기하는데 김현한이 10월 24일 날 하요일날 당직이는데 혹시 10월 11일요 다음 주 10월 날이 내 당직이 저 1점 켜보고 바로 다시 전화 드릴게요"""
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
