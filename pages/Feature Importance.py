import streamlit as st
import numpy as np
import pickle
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

st.sidebar.markdown("# Feature Importance")

# XGBoost 모델 로드
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# 특성 변수 이름 가져오기
feature_names = xgb_model.get_booster().feature_names

# 제외할 특성 변수 설정 (고정값인 average_temp와 average_humidity)
exclude_features = ['average_temp', 'average_humidity']

# 제외할 특성 변수를 제외하고 남은 특성 변수만 선택
remaining_features = [feat for feat in feature_names if feat not in exclude_features]

# XGBoost 모델의 feature importance 추출 (제외된 특성 변수 제외)
importance = xgb_model.feature_importances_
filtered_importance = [importance[i] for i, feat in enumerate(feature_names) if feat in remaining_features]

# 중요도를 내림차순으로 정렬하여 상위 5개 특성 변수 선택
sorted_idx = (-1 * np.array(filtered_importance)).argsort()[:5]
top_features = [remaining_features[i] for i in sorted_idx]
top_importance = [filtered_importance[i] for i in sorted_idx]

# Streamlit 애플리케이션 정의
st.title('Feature Importance Top 5')

# 상위 5개 특성 변수의 Feature Importance 시각화 (오름차순)
fig, ax = plt.subplots()
ax.barh(top_features[::-1], top_importance[::-1])  # 역순으로 출력
ax.set_xlabel('Feature Importance')
ax.set_title('Top 5 Feature Importance')
st.pyplot(fig)