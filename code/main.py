import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="화이팅 - 공부 시간과 시험 성적 탐구", layout="wide")

# 상단 팀 소개 및 탐구 주제 헤더
st.title("📊 공부 시간은 정말 시험 성적에 영향을 미치는가?")
st.markdown("**팀 화이팅** | 21010 정아영 · 21001 김슬기 · 21007 이다빈")

with st.expander("📌 우리가 이 주제를 선정한 이유 (탐구 배경)", expanded=True):
    st.markdown(
        """
        - **21010 정아영:** 학생들의 공부 시간과 시험 성적을 수치로 비교하여 두 변수 사이에 어떤 관계가 있는지 객관적으로 알아볼 수 있기 때문이다.
        - **21001 김슬기:** 평소 공부를 많이 하면 시험 성적도 높아지는지 궁금했기 때문이다.
        - **21007 이다빈:** 공부 시간과 성적의 관계를 분석하면 효율적인 공부 습관을 만드는 데 도움이 될 것이라고 생각했기 때문이다.
        """
    )

st.divider()

# 기본 설문조사 데이터 (접속하자마자 바로 그래프가 보이도록 구성)
INITIAL_SURVEY_DATA = [
    {"응답자": "학생 1", "하루 평균 공부 시간(시간)": 0.5, "시험 평균 점수(점)": 62, "주된 공부 방법": "교과서·프린트 복습"},
    {"응답자": "학생 2", "하루 평균 공부 시간(시간)": 1.0, "시험 평균 점수(점)": 68, "주된 공부 방법": "문제집 풀이"},
    {"응답자": "학생 3", "하루 평균 공부 시간(시간)": 1.0, "시험 평균 점수(점)": 71, "주된 공부 방법": "인강·인터넷 강의"},
    {"응답자": "학생 4", "하루 평균 공부 시간(시간)": 1.5, "시험 평균 점수(점)": 74, "주된 공부 방법": "문제집 풀이"},
    {"응답자": "학생 5", "하루 평균 공부 시간(시간)": 1.5, "시험 평균 점수(점)": 69, "주된 공부 방법": "오답노트 정리"},
    {"응답자": "학생 6", "하루 평균 공부 시간(시간)": 2.0, "시험 평균 점수(점)": 78, "주된 공부 방법": "문제집 풀이"},
    {"응답자": "학생 7", "하루 평균 공부 시간(시간)": 2.0, "시험 평균 점수(점)": 82, "주된 공부 방법": "교과서·프린트 복습"},
    {"응답자": "학생 8", "하루 평균 공부 시간(시간)": 2.5, "시험 평균 점수(점)": 81, "주된 공부 방법": "오답노트 정리"},
    {"응답자": "학생 9", "하루 평균 공부 시간(시간)": 2.5, "시험 평균 점수(점)": 85, "주된 공부 방법": "문제집 풀이"},
    {"응답자": "학생 10", "하루 평균 공부 시간(시간)": 3.0, "시험 평균 점수(점)": 84, "주된 공부 방법": "교과서·프린트 복습"},
    {"응답자": "학생 11", "하루 평균 공부 시간(시간)": 3.0, "시험 평균 점수(점)": 89, "주된 공부 방법": "오답노트 정리"},
    {"응답자": "학생 12", "하루 평균 공부 시간(시간)": 3.5, "시험 평균 점수(점)": 88, "주된 공부 방법": "문제집 풀이"},
    {"응답자": "학생 13", "하루 평균 공부 시간(시간)": 3.5, "시험 평균 점수(점)": 92, "주된 공부 방법": "교과서·프린트 복습"},
    {"응답자": "학생 14", "하루 평균 공부 시간(시간)": 4.0, "시험 평균 점수(점)": 91, "주된 공부 방법": "오답노트 정리"},
    {"응답자": "학생 15", "하루 평균 공부 시간(시간)": 4.0, "시험 평균 점수(점)": 95, "주된 공부 방법": "문제집 풀이"},
    {"응답자": "학생 16", "하루 평균 공부 시간(시간)": 4.5, "시험 평균 점수(점)": 90, "주된 공부 방법": "인강·인터넷 강의"},
    {"응답자": "학생 17", "하루 평균 공부 시간(시간)": 5.0, "시험 평균 점수(점)": 94, "주된 공부 방법": "오답노트 정리"},
    {"응답자": "학생 18", "하루 평균 공부 시간(시간)": 5.5, "시험 평균 점수(점)": 96, "주된 공부 방법": "문제집 풀이"},
]

if "survey_df" not in st.session_state:
    st.session_state.survey_df = pd.DataFrame(INITIAL_SURVEY_DATA)

# 1. 실시간 설문조사 참여 폼
st.markdown("### 1. 설문조사 참여하기 (실시간 데이터 추가)")
st.caption("아래 설문에 응답하고 '설문 제출하기'를 누르면 아래 분석 결과와 그래프가 즉시 업데이트됩니다.")

with st.form("survey_form", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns([1.2, 1.5, 1.5, 1.8])
    with c1:
        name = st.text_input("이름 또는 별명", placeholder="예: 학생 19")
    with c2:
        study_hours = st.number_input("평소 하루 평균 공부 시간 (시간)", min_value=0.0, max_value=15.0, value=2.5, step=0.5)
    with c3:
        exam_score = st.number_input("최근 시험 평균 점수 (점)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
    with c4:
        study_method = st.selectbox(
            "가장 효과가 좋았던 공부 방법",
            ["문제집 풀이", "교과서·프린트 복습", "오답노트 정리", "인강·인터넷 강의", "친구와 질문·토론"]
        )
    submitted = st.form_submit_button("✅ 설문 제출하기")

    if submitted:
        respondent_name = name.strip() if name.strip() else f"학생 {len(st.session_state.survey_df) + 1}"
        new_row = pd.DataFrame([{
            "응답자": respondent_name,
            "하루 평균 공부 시간(시간)": float(study_hours),
            "시험 평균 점수(점)": float(exam_score),
            "주된 공부 방법": study_method
        }])
        st.session_state.survey_df = pd.concat([st.session_state.survey_df, new_row], ignore_index=True)
        st.success(f"'{respondent_name}'님의 설문 응답이 반영되었습니다!")

df = st.session_state.survey_df.copy()

# 통계 계산 (상관계수 및 회귀직선)
x = df["하루 평균 공부 시간(시간)"].values
y = df["시험 평균 점수(점)"].values

corr = np.corrcoef(x, y)[0, 1] if len(df) > 1 else 0.0
slope, intercept = np.polyfit(x, y, 1) if len(df) > 1 else (0.0, np.mean(y))

# 2. 핵심 분석 결과 요약
st.markdown("### 2. 설문조사 핵심 분석 결과")
m1, m2, m3, m4 = st.columns(4)
m1.metric("총 설문 참여 인원", f"{len(df)}명")
m2.metric("평균 공부 시간", f"{np.mean(x):.2f}시간")
m3.metric("상관계수 (r)", f"{corr:.3f}")
m4.metric("회귀 기울기", f"+{slope:.2f}점/시간" if slope >= 0 else f"{slope:.2f}점/시간")

if corr >= 0.7:
    corr_desc = "강한 양의 상관관계"
elif corr >= 0.3:
    corr_desc = "뚜렷한 양의 상관관계"
elif corr >= 0.1:
    corr_desc = "약한 양의 상관관계"
elif corr <= -0.1:
    corr_desc = "음의 상관관계"
else:
    corr_desc = "뚜렷하지 않은 상관관계"

st.info(
    f"분석 결과, 공부 시간과 시험 성적 사이에는 **{corr_desc}(r = {corr:.3f})**가 나타났습니다. "
    f"추세선 기준으로 하루 공부 시간이 **1시간 늘어날 때 시험 성적은 평균 약 {slope:.2f}점** 변화하는 경향을 보였습니다."
)

# 3. 산점도 및 회귀직선 그래프
st.markdown("### 3. 공부 시간과 시험 성적의 관계 그래프")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["하루 평균 공부 시간(시간)"],
    y=df["시험 평균 점수(점)"],
    mode="markers+text",
    text=df["응답자"],
    textposition="top center",
    name="설문 응답자",
    marker=dict(size=10, color="#2E86C1", opacity=0.8),
    hovertemplate="<b>%{text}</b><br>공부 시간: %{x}시간<br>시험 성적: %{y}점<extra></extra>"
))

x_line = np.linspace(max(0, x.min() - 0.3), x.max() + 0.3, 100)
y_line = slope * x_line + intercept
fig.add_trace(go.Scatter(
    x=x_line,
    y=y_line,
    mode="lines",
    name="추세선 (회귀직선)",
    line=dict(color="#E74C3C", width=3, dash="dash")
))

fig.update_layout(
    title="하루 평균 공부 시간(시간)에 따른 시험 평균 점수(점)",
    xaxis_title="하루 평균 공부 시간 (시간)",
    yaxis_title="시험 평균 점수 (점)",
    yaxis=dict(range=[0, 105]),
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# 4. 공부 시간 구간별 & 공부 방법별 비교
st.markdown("### 4. 공부 시간 구간별 · 공부 방법별 평균 성적 비교")
col_left, col_right = st.columns(2)

with col_left:
    df["공부 시간 구간"] = pd.cut(
        df["하루 평균 공부 시간(시간)"],
        bins=[-0.1, 1.5, 3.0, 4.5, 24.0],
        labels=["1.5시간 이하", "1.5시간 초과 ~ 3시간", "3시간 초과 ~ 4.5시간", "4.5시간 초과"]
    )
    group_time = df.groupby("공부 시간 구간", observed=False)["시험 평균 점수(점)"].agg(["count", "mean"]).reset_index()
    group_time.columns = ["공부 시간 구간", "학생 수(명)", "평균 점수(점)"]
    group_time["평균 점수(점)"] = group_time["평균 점수(점)"].fillna(0).round(1)

    bar1 = go.Figure(go.Bar(
        x=group_time["공부 시간 구간"].astype(str),
        y=group_time["평균 점수(점)"],
        text=group_time["평균 점수(점)"],
        textposition="outside",
        marker_color="#5DADE2"
    ))
    bar1.update_layout(
        title="공부 시간 구간별 평균 점수",
        xaxis_title="하루 공부 시간 구간",
        yaxis_title="평균 점수 (점)",
        yaxis=dict(range=[0, 110]),
        height=380
    )
    st.plotly_chart(bar1, use_container_width=True)

with col_right:
    group_method = df.groupby("주된 공부 방법")["시험 평균 점수(점)"].agg(["count", "mean"]).reset_index()
    group_method.columns = ["주된 공부 방법", "학생 수(명)", "평균 점수(점)"]
    group_method["평균 점수(점)"] = group_method["평균 점수(점)"].round(1)

    bar2 = go.Figure(go.Bar(
        x=group_method["주된 공부 방법"],
        y=group_method["평균 점수(점)"],
        text=group_method["평균 점수(점)"],
        textposition="outside",
        marker_color="#48C9B0"
    ))
    bar2.update_layout(
        title="주된 공부 방법별 평균 점수",
        xaxis_title="공부 방법",
        yaxis_title="평균 점수 (점)",
        yaxis=dict(range=[0, 110]),
        height=380
    )
    st.plotly_chart(bar2, use_container_width=True)

# 5. 전체 설문조사 데이터 표 및 다운로드
st.markdown("### 5. 전체 설문조사 응답 데이터 확인 및 다운로드")
st.dataframe(df.drop(columns=["공부 시간 구간"], errors="ignore"), use_container_width=True, hide_index=True)

csv_data = df.drop(columns=["공부 시간 구간"], errors="ignore").to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button(
    label="📥 설문조사 결과 CSV 파일로 내려받기",
    data=csv_data,
    file_name="team_fighting_study_survey.csv",
    mime="text/csv"
)

# 6. 탐구 결론 및 주의할 점
st.markdown("### 6. 탐구 결론 및 해석 시 주의할 점")
st.markdown(
    """
    - **탐구 결론:** 설문조사 데이터를 분석한 결과, 전반적으로 하루 평균 공부 시간이 길수록 시험 평균 점수가 높아지는 경향을 확인할 수 있습니다.
    - **주의할 점 (상관관계 ≠ 인과관계):** 공부 시간 외에도 수업 집중도, 기초 학습량, 수면 시간, 공부 방법의 효율성 등 다양한 요인이 시험 성적에 영향을 미칠 수 있으므로, 단순히 시간만 늘리는 것보다 자신에게 맞는 효율적인 학습 전략을 세우는 것이 중요합니다.
    """
)
