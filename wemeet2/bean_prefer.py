import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")

# 모델 로드
c_model = joblib.load("./model/Kmeans_model.joblib")
rf_model = joblib.load("./model/모든 데이터 학습_best_rf_model.joblib")

# 데이터 로드
data = pd.read_csv("./model/원두와 5가지 지표.csv")
data.set_index("Bean", inplace=True)
cosine_sim = cosine_similarity(data)
cosine_sim_df = pd.DataFrame(cosine_sim, index=data.index, columns=data.index)

brand_names = ["TheVenti", "Mega", "Paik", "Starbucks", "Ediya", "Compose", "Twosome"]

# 세션 초기화
if 'dislike_list' not in st.session_state:
    st.session_state.dislike_list = []
if 'liked_beans' not in st.session_state:
    st.session_state.liked_beans = []
if 'recommended_beans' not in st.session_state:
    st.session_state.recommended_beans = []
if 'final_recommendations' not in st.session_state:
    st.session_state.final_recommendations = []


# 추천 평가 함수

def recommend_beans(purchased_bean):
    return list(
        cosine_sim_df[purchased_bean]
        .sort_values(ascending=False)
        .drop([purchased_bean] + brand_names + st.session_state.dislike_list, axis=0)
        .head(3).index
    )


# 추천 평가

def evaluate_recommendations(base_bean):
    sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
    user_feedback = {}

    if not st.session_state.recommended_beans:
        st.session_state.recommended_beans = recommend_beans(base_bean)

    st.write("### 추천 원두 리스트:")

    for bean in st.session_state.recommended_beans:
        selected = st.feedback("thumbs", key=bean)
        if selected is not None:
            st.markdown(f"{bean}: {sentiment_mapping[selected]}")
            user_feedback[bean] = selected

    if st.button("평가 완료"):
        for bean, feedback in user_feedback.items():
            if feedback == 0:  # thumb_down
                if bean not in st.session_state.dislike_list:
                    st.session_state.dislike_list.append(bean)
                    st.session_state.recommended_beans.remove(bean)
                    new_bean = recommend_beans(base_bean)
                    for nb in new_bean:
                        if nb not in st.session_state.recommended_beans and nb not in st.session_state.dislike_list:
                            st.session_state.recommended_beans.append(nb)
                            break
            else:  # thumb_up
                if bean not in st.session_state.liked_beans:
                    st.session_state.liked_beans.append(bean)

        if len(st.session_state.liked_beans) == 3:
            st.session_state.final_recommendations = st.session_state.liked_beans
            st.write("### 추천 원두 항목 리스트:")
            st.write(st.session_state.final_recommendations)

    if st.button("다시 시작"):
        st.session_state.dislike_list = []
        st.session_state.liked_beans = []
        st.session_state.recommended_beans = []
        st.session_state.final_recommendations = []
        st.write("추천 시스템이 초기화되었습니다.")

    if st.button("페이지 가기"):
        st.markdown("[원두 구매하러 가기](https://www.wonderroom.co.kr/)")


# UI 구성
st.title("커피 원두 추천 시스템")

purchase_history = st.radio("원더룸에서 원두를 구입해 본 적이 있습니까?", ["예", "아니오"])

exclude_beans = ["TheVenti", "Mega", "Paik", "Starbucks", "Ediya", "Compose", "Twosome",
                 "Ethiopia Yirgacheffe Kochere Washed"]

if purchase_history == "예":
    purchased_bean = st.selectbox(
        "구입했던 원두 중 선호한 원두를 선택해주세요",
        [bean for bean in data.index if bean not in exclude_beans]
    )

    if st.button("추천 원두 확인"):
        st.session_state.recommended_beans = recommend_beans(purchased_bean)

    if st.session_state.recommended_beans:
        for i, bean in enumerate(st.session_state.recommended_beans, start=1):
            st.write(f"{i}. {bean}")
        evaluate_recommendations(purchased_bean)

else:
    sex = st.radio("성별을 선택하세요", ["남", "여"])
    age = st.slider("나이를 입력하세요", 18, 60, 25)
    is_student = st.radio("직업을 선택하세요", ["학생", "기타"])
    frequency = st.selectbox("커피를 마시는 빈도", ["매일", "주 5-6회", "주 3-4회", "주 2회", "주 1회 미만"])
    method = st.selectbox("커피 내리는 방법", ["에스프레소 머신", "핸드 드립", "커피메이커", "콜드브루"])
    coffee_type = st.selectbox("커피 타입", ["블랙", "우유 라떼", "시럽 커피", "설탕 커피"])
    flavor = st.selectbox("커피 풍미", ["고소한, 구운", "달콤, 설탕", "초콜릿", "과일", "꽃향"])

    if st.button("추천 원두 확인"):
        x = [
            1 if sex == "남" else 0, age, 1 if is_student == "학생" else 0,
            9 if frequency == "매일" else 7 if frequency == "주 5-6회" else 5 if frequency == "주 3-4회" else 3 if frequency == "주 2회" else 1,
            4 if method == "에스프레소 머신" else 3 if method == "핸드 드립" else 2 if method == "커피메이커" else 1,
            4 if coffee_type == "블랙" else 3 if coffee_type == "우유 라떼" else 2 if coffee_type == "시럽 커피" else 1,
            5 if flavor == "고소한, 구운" else 4 if flavor == "달콤, 설탕" else 3 if flavor == "초콜릿" else 2 if flavor == "과일" else 1
        ]
        cluster_prediction = c_model.predict(np.array(x).reshape(1, -1))[0]
        x.append(cluster_prediction)
        cafe_prediction = rf_model.predict(np.array(x).reshape(1, -1))[0]
        predicted_cafe = brand_names[cafe_prediction]
        st.session_state.recommended_beans = recommend_beans(predicted_cafe)

        if st.session_state.recommended_beans:
            for i, bean in enumerate(st.session_state.recommended_beans, start=1):
                st.write(f"{i}. {bean}")
            evaluate_recommendations(predicted_cafe)

