import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="공정 시뮬레이터", page_icon="🔧", layout="wide", initial_sidebar_state="expanded")
st.sidebar.markdown("# 공정 시뮬레이터 🎈")

# Streamlit 애플리케이션 설정

def data_read(data):
    df_ = pd.read_csv(f"./{data}")
    return df_

def data_split(data):
    # 데이터를 train과 나머지로 분할
    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)

    # 나머지 데이터를 validation과 test로 분할
    validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # 분할된 데이터셋의 크기 확인
    # print("Train 데이터 크기:", train_data.shape)
    # print("Validation 데이터 크기:", validation_data.shape)
    # print("Test 데이터 크기:", test_data.shape)

    # train_data에서 특성 변수와 클래스 레이블 분리
    X_train_ = train_data.drop('label', axis=1)  # 특성 변수
    y_train_ = train_data['label']               # 클래스 레이블

    # validation_data에서 특성 변수와 클래스 레이블 분리
    X_validation_ = validation_data.drop('label', axis=1)  # 특성 변수
    y_validation_ = validation_data['label']               # 클래스 레이블

    # test_data에서 특성 변수와 클래스 레이블 분리
    X_test_ = test_data.drop('label', axis=1)  # 특성 변수
    y_test_ = test_data['label']               # 클래스 레이블

    return X_train_,y_train_, X_validation_, y_validation_, X_test_, y_test_

def pickle_load(model):
    # 저장된 모델 로드
    with open(model, 'rb') as x:
        xgb_model_ = pickle.load(x)
        return xgb_model_
    # 모델 로드 예시
    # xgb_model = load('xgb_model.joblib')
    # mlp_model = load('mlp_model.joblib')

df = data_read('processed_data.csv')

X_test = data_split(df)[4]

xgb_model = pickle_load('xgb_model.pkl')

# 예측 예시
xgb_prediction = xgb_model.predict(X_test)

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

#불량조건에서 최소한의 양품조건으로 되는 수치
# stage3_density_deviation: -1.8 => -1.3 이상
# stage3_o2_deviation: -0.2
# stage3_co2_deviation: 1.7 => -1 이하
# stage4_flow_deviation: 0.1 => 1.4 이상
# stage4_density_deviation: -3.3 => -1.3 이상

#불량조건
# stage3_density_deviation: -1.8
# stage3_o2_deviation: -0.2
# stage3_co2_deviation: 1.7
# stage4_flow_deviation: 0.1
# stage4_density_deviation: -3.3


# 스타일을 적용할 커스텀 CSS 코드
custom_css = """
<style>
div[data-baseweb="input"] {
    width: 300px; /* 텍스트 입력 상자의 가로 길이 */
}

div[data-baseweb="input"] input {
    width: 300px; /* 텍스트 입력 상자의 가로 길이 */
    background-color: #fffaf0; /* 배경색 */
    border: 1px solid #ccc; /* 테두리 스타일 */
    font-size: 16px; /* 폰트 크기 */
}
div[role="alert"] {
    width: 300px;
}
</style>
"""

# 커스텀 CSS 코드 적용
st.markdown(custom_css, unsafe_allow_html=True)

# stage3_o2_deviation 입력값 처리
try:
    stage3_density_deviation = st.text_input('stage3_density_deviation', '0',key="text_input1")
    if stage3_density_deviation.strip():  # 공백이 아닌 경우에만 float로 변환
        stage3_density_deviation = float(stage3_density_deviation)
    else:
        stage3_density_deviation = 0.0  # 공백인 경우 기본값으로 설정
except ValueError:
    st.error('숫자로 입력해주세요')
    
# stage3_o2_deviation 입력값 처리
try:
    stage3_o2_deviation = st.text_input('stage3_o2_deviation', '0', key="text_input2")
    if stage3_o2_deviation.strip():  # 공백이 아닌 경우에만 float로 변환
        stage3_o2_deviation = float(stage3_o2_deviation)
    else:
        stage3_o2_deviation = 0.0  # 공백인 경우 기본값으로 설정
except ValueError:
    st.error('숫자로 입력해주세요')
    
# stage3_co2_deviation 입력값 처리
try:
    stage3_co2_deviation = st.text_input('stage3_co2_deviation', '0',key="text_input3")
    if stage3_co2_deviation.strip():  # 공백이 아닌 경우에만 float로 변환
        stage3_co2_deviation = float(stage3_co2_deviation)
    else:
        stage3_co2_deviation = 0.0  # 공백인 경우 기본값으로 설정
except ValueError:
    st.error('숫자로 입력해주세요')
    
# stage4_flow_deviation 입력값 처리
try:
    stage4_flow_deviation = st.text_input('stage4_flow_deviation', '0',key="text_input4")
    if stage4_flow_deviation.strip():  # 공백이 아닌 경우에만 float로 변환
        stage4_flow_deviation = float(stage4_flow_deviation)
    else:
        stage4_flow_deviation = 0.0  # 공백인 경우 기본값으로 설정
except ValueError:
    st.error('숫자로 입력해주세요')
    
# stage4_density_deviation 입력값 처리
try:
    stage4_density_deviation = st.text_input('stage4_density_deviation', '0',key="text_input5")
    if stage4_density_deviation.strip():  # 공백이 아닌 경우에만 float로 변환
        stage4_density_deviation = float(stage4_density_deviation)
    else:
        stage4_density_deviation = 0.0  # 공백인 경우 기본값으로 설정
except ValueError:
    st.error('숫자로 입력해주세요')

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
if st.button('예측하기', key='predict_button', help='XGBoost 모델을 사용하여 예측을 수행합니다.'):
    try:
        input_df = pd.DataFrame([input_data])
        prediction = xgb_model.predict(input_df)

        # 예측 결과 출력
        st.subheader('예측 결과')
        if prediction[0] == 0:
            st.write('양품입니다!')
        else:
            st.write('불량입니다.')
    except ValueError:
        st.error('입력값을 다시 확안해주세요')