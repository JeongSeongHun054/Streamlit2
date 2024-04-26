import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from main import data_read, data_split, pickle_load

# 경고 메시지 숨기기
st.set_option('deprecation.showPyplotGlobalUse', False)

st.sidebar.markdown("# Visualization")

df = data_read('processed_data.csv')

X_test = data_split(df)[4]

xgb_model = pickle_load('xgb_model.pkl')

# 예측 예시
xgb_prediction = xgb_model.predict(X_test)

# 예측 결과를 상세히 보고 싶을 경우, 예측 클래스별 개수 표시
xgb_prediction_counts = pd.Series(xgb_prediction).value_counts()
st.write("#### 예측 클래스별 개수")
st.write(xgb_prediction_counts)

# 예측 결과를 양품과 불량으로 매핑
xgb_prediction_label = ['fair' if pred == 0 else 'error' for pred in xgb_prediction]

# 예측 결과를 데이터프레임으로 변환
xgb_result_df = pd.DataFrame({'Prediction': xgb_prediction_label})

# 양품과 불량의 비율 계산
xgb_counts = xgb_result_df['Prediction'].value_counts(normalize=True)

# 비율을 퍼센트로 변환하여 출력
st.write("### XGBoost 예측 결과 비율")
for label, count in xgb_counts.items():
    st.write(f"{label}: {count*100:.2f}%")

# 예측 클래스별 개수 시각화
plt.bar(xgb_prediction_counts.index, xgb_prediction_counts.values, color=['skyblue','salmon'])
plt.xlabel('Status')
plt.ylabel('Count')
plt.title('XGBoost Prediction Class Counts')

# 그래프에 클래스 이름 표시
for i, count in enumerate(xgb_prediction_counts.values):
    plt.text(i, count + 1, str(count), ha='center', va='bottom', fontsize=12)

st.pyplot()

# 예측 결과 비율 시각화
fig, ax = plt.subplots()
ax.pie(xgb_counts.values, labels=xgb_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightskyblue', 'lightcoral'])
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)