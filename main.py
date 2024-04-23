import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# print('sklearn 버전:',sklearn.__version__)
# print('joblib 버전:',joblib.__version__)

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")

df = pd.read_csv("./processed_data.csv")

# 데이터를 train과 나머지로 분할
train_data, temp_data = train_test_split(df, test_size=0.4, random_state=42)

# 나머지 데이터를 validation과 test로 분할
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# 분할된 데이터셋의 크기 확인
# print("Train 데이터 크기:", train_data.shape)
# print("Validation 데이터 크기:", validation_data.shape)
# print("Test 데이터 크기:", test_data.shape)

# train_data에서 특성 변수와 클래스 레이블 분리
X_train = train_data.drop('label', axis=1)  # 특성 변수
y_train = train_data['label']               # 클래스 레이블

# validation_data에서 특성 변수와 클래스 레이블 분리
X_validation = validation_data.drop('label', axis=1)  # 특성 변수
y_validation = validation_data['label']               # 클래스 레이블

# test_data에서 특성 변수와 클래스 레이블 분리
X_test = test_data.drop('label', axis=1)  # 특성 변수
y_test = test_data['label']               # 클래스 레이블

# 저장된 모델 로드
with open('xgb_model.pkl', 'rb') as x:
    xgb_model = pickle.load(x)

# 모델 로드 예시
# xgb_model = load('xgb_model.joblib')
# mlp_model = load('mlp_model.joblib')

# 예측 예시
xgb_prediction = xgb_model.predict(X_test)

# 예측 결과를 상세히 보고 싶을 경우, 예측 클래스별 개수 표시
xgb_prediction_counts = pd.Series(xgb_prediction).value_counts()
st.write("#### 예측 클래스별 개수")
st.write(xgb_prediction_counts)

# 예측 결과를 양품과 불량으로 매핑
xgb_prediction_label = ['양품' if pred == 0 else '불량' for pred in xgb_prediction]

# 예측 결과를 데이터프레임으로 변환
xgb_result_df = pd.DataFrame({'Prediction': xgb_prediction_label})

# 양품과 불량의 비율 계산
xgb_counts = xgb_result_df['Prediction'].value_counts(normalize=True)

# 비율을 퍼센트로 변환하여 출력
st.write("### XGBoost 예측 결과 비율")
for label, count in xgb_counts.items():
    st.write(f"{label}: {count*100:.2f}%")

# Streamlit 애플리케이션 정의
st.title('공정 시뮬레이터')

# 사용자 입력 컴포넌트 정의
average_humidity = 73
average_temp = 30 
stage1_flow_deviation = 0.01
stage1_density_deviation = 0.24
stage1_viscosity_deviation = -0.8
stage1_co2_deviation = 0.24
stage1_o2_deviation = -0.008
stage1_n_deviation = -0.001
stage2_flow_deviation = -0.48
stage2_density_deviation = 0.42
stage2_viscosity_deviation = 0.03
stage2_co2_deviation = -0.008
stage2_o2_deviation = -0.006
stage2_n_deviation = -0.8
stage3_flow_deviation = 0.7
stage3_viscosity_deviation = -0.01
stage3_n_deviation = -0.8
stage4_flow_deviation = 0.17
stage4_viscosity_deviation = -0.48
stage4_co2_deviation = -0.12
stage4_o2_deviation = -0.74
stage4_n_deviation = -0.01
stage5_flow_deviation = -0.74
stage5_density_deviation = -0.55
stage5_viscosity_deviation = 0.78
stage5_co2_deviation = -0.02
stage5_o2_deviation = 0.01
stage5_n_deviation = -0.03

#불량조건
# stage3_density_deviation: -1.8
# stage3_o2_deviation: -0.2
# stage3_co2_deviation: 1.7
# stage4_flow_deviation: 0.1
# stage4_density_deviation: -3.3

# stage3_density_deviation 입력값 처리
stage3_density_deviation = st.text_input('stage3_density_deviation', '0')
if stage3_density_deviation.strip():  # 공백이 아닌 경우에만 float로 변환
    stage3_density_deviation = float(stage3_density_deviation)
else:
    stage3_density_deviation = 0.0  # 공백인 경우 기본값으로 설정

# stage3_o2_deviation 입력값 처리
stage3_o2_deviation = st.text_input('stage3_o2_deviation', '0')
if stage3_o2_deviation.strip():  # 공백이 아닌 경우에만 float로 변환
    stage3_o2_deviation = float(stage3_o2_deviation)
elif type(stage3_o2_deviation)=='String':
    alert
else:
    stage3_o2_deviation = 0.0  # 공백인 경우 기본값으로 설정

# stage3_co2_deviation 입력값 처리
stage3_co2_deviation = st.text_input('stage3_co2_deviation', '0')
if stage3_co2_deviation.strip():  # 공백이 아닌 경우에만 float로 변환
    stage3_co2_deviation = float(stage3_co2_deviation)
else:
    stage3_co2_deviation = 0.0  # 공백인 경우 기본값으로 설정

# stage4_flow_deviation 입력값 처리
stage4_flow_deviation = st.text_input('stage4_flow_deviation', '0')
if stage4_flow_deviation.strip():  # 공백이 아닌 경우에만 float로 변환
    stage4_flow_deviation = float(stage4_flow_deviation)
else:
    stage4_flow_deviation = 0.0  # 공백인 경우 기본값으로 설정

# stage4_density_deviation 입력값 처리
stage4_density_deviation = st.text_input('stage4_density_deviation', '0')
if stage4_density_deviation.strip():  # 공백이 아닌 경우에만 float로 변환
    stage4_density_deviation = float(stage4_density_deviation)
else:
    stage4_density_deviation = 0.0  # 공백인 경우 기본값으로 설정

# stage3_density_deviation = float(st.text_input('stage3_density_deviation'))
# stage3_o2_deviation = float(st.text_input('stage3_o2_deviation_variable'))
# stage3_co2_deviation = float(st.text_input('stage3_co2_deviation'))
# stage4_flow_deviation = float(st.text_input('stage4_flow_deviation'))
# stage4_density_deviation = float(st.text_input('stage4_density_deviation'))

# stage3_density_deviation = st.slider('stage3_density_deviation', min_value=0, max_value=100, value=50)
# stage3_o2_deviation = st.slider('stage3_o2_deviation_variable', min_value=0, max_value=100, value=50)
# stage3_co2_deviation = st.slider('stage3_co2_deviation', min_value=0, max_value=100, value=50)
# stage4_flow_deviation = st.slider('stage4_flow_deviation', min_value=0, max_value=100, value=50)
# stage4_density_deviation = st.slider('stage4_density_deviation', min_value=0, max_value=100, value=50)

# 사용자 입력으로 XGBoost 모델에 전달하여 예측 수행
input_data = {    
    'average_temp': average_temp,
    'average_humidity': average_humidity, 
    'stage1_flow_deviation': stage1_flow_deviation,
    'stage1_density_deviation':stage1_density_deviation,
    'stage1_viscosity_deviation':stage1_viscosity_deviation,
    'stage1_co2_deviation':stage1_co2_deviation,
    'stage1_o2_deviation':stage1_o2_deviation,
    'stage1_n_deviation':stage1_n_deviation,
    'stage2_flow_deviation':stage2_flow_deviation,
    'stage2_density_deviation':stage2_density_deviation,
    'stage2_viscosity_deviation':stage2_viscosity_deviation,
    'stage2_co2_deviation':stage2_co2_deviation,
    'stage2_o2_deviation':stage2_o2_deviation,
    'stage2_n_deviation':stage2_n_deviation,
    'stage3_flow_deviation':stage3_flow_deviation,
    'stage3_density_deviation': stage3_density_deviation,
    'stage3_viscosity_deviation' : stage3_viscosity_deviation,
    'stage3_co2_deviation': stage3_co2_deviation,
    'stage3_o2_deviation': stage3_o2_deviation,
    'stage3_n_deviation' : stage3_n_deviation,
    'stage4_flow_deviation' : stage4_flow_deviation,
    'stage4_density_deviation': stage4_density_deviation,
    'stage4_viscosity_deviation' : stage4_viscosity_deviation,
    'stage4_co2_deviation' : stage4_co2_deviation,
    'stage4_o2_deviation' : stage4_o2_deviation,
    'stage4_n_deviation' : stage4_n_deviation,
    'stage5_flow_deviation' : stage5_flow_deviation,
    'stage5_density_deviation' : stage5_density_deviation,
    'stage5_viscosity_deviation' : stage5_viscosity_deviation,
    'stage5_co2_deviation' : stage5_co2_deviation,
    'stage5_o2_deviation' : stage5_o2_deviation,
    'stage5_n_deviation' : stage5_n_deviation
}
if st.button('예측하기'):
    input_df = pd.DataFrame([input_data])
    prediction = xgb_model.predict(input_df)

    # 예측 결과 출력
    st.subheader('예측 결과')
    if prediction[0] == 0:
        st.write('양품입니다!')
    else:
        st.write('불량입니다.')

# 시각화 및 결과 출력

# 추가적인 시각화 또는 결과 확인을 위한 코드 작성