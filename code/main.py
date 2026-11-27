import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="화이팅 - 설문조사 및 성적 탐구", layout="wide")

# 상단 헤더
st.title("📝 공부 시간은 정말 시험 성적에 영향을 미치는가?")
st.markdown("**팀 화이팅** (21010 정아영 · 21001 김슬기 · 21007 이다빈) | **학생 설문조사 및 데이터 분석**")
st.divider()

# 기본 수집된 설문 데이터 (현재까지 응답한 18명)
INITIAL_DATA = [
    {"타임스탬프": "2026-11-26 14:10", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 0.5, "시험 평균 점수(점)": 62, "주된 공부 방법": "교과서·프린트 복습"},
    {"타임스탬프": "2026-11-26 14:12", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 1.0, "시험 평균 점수(점)": 68, "주된 공부 방법": "문제집 풀이"},
    {"타임스탬프": "2026-11-26 14:15", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 1.0, "시험 평균 점수(점)": 71, "주된 공부 방법": "인강·인터넷 강의"},
    {"타임스탬프": "2026-11-26 14:18", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 1.5, "시험 평균 점수(점)": 74, "주된 공부 방법": "문제집 풀이"},
    {"타임스탬프": "2026-11-26 14:21", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 1.5, "시험 평균 점수(점)": 69, "주된 공부 방법": "오답노트 정리"},
    {"타임스탬프": "2026-11-26 14:25", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 2.0, "시험 평균 점수(점)": 78, "주된 공부 방법": "문제집 풀이"},
    {"타임스탬프": "2026-11-26 14:30", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 2.0, "시험 평균 점수(점)": 82, "주된 공부 방법": "교과서·프린트 복습"},
    {"타임스탬프": "2026-11-26 14:33", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 2.5, "시험 평균 점수(점)": 81, "주된 공부 방법": "오답노트 정리"},
    {"타임스탬프": "2026-11-26 14:40", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 2.5, "시험 평균 점수(점)": 85, "주된 공부 방법": "문제집 풀이"},
    {"타임스탬프": "2026-11-26 15:02", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 3.0, "시험 평균 점수(점)": 84, "주된 공부 방법": "교과서·프린트 복습"},
    {"타임스탬프": "2026-11-26 15:10", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 3.0, "시험 평균 점수(점)": 89, "주된 공부 방법": "오답노트 정리"},
    {"타임스탬프": "2026-11-26 15:14", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 3.5, "시험 평균 점수(점)": 88, "주된 공부 방법": "문제집 풀이"},
    {"타임스탬프": "2026-11-26 15:20", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 3.5, "시험 평균 점수(점)": 92, "주된 공부 방법": "교과서·프린트 복습"},
    {"타임스탬프": "2026-11-26 15:25", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 4.0, "시험 평균 점수(점)": 91, "주된 공부 방법": "오답노트 정리"},
    {"타임스탬프": "2026-11-26 15:31", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 4.0, "시험 평균 점수(점)": 95, "주된 공부 방법": "문제집 풀이"},
    {"타임스탬프": "2026-11-26 15:40", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 4.5, "시험 평균 점수(점)": 90, "주된 공부 방법": "인강·인터넷 강의"},
    {"타임스탬프": "2026-11-26 15:45", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 5.0, "시험 평균 점수(점)": 94, "주된 공부 방법": "오답노트 정리"},
    {"타임스탬프": "2026-11-26 15:50", "학년/반": "2학년", "하루 평균 공부 시간(시간)": 5.5, "시험 평균 점수(점)": 96, "주된 공부 방법": "문제집 풀이"},
]

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(INITIAL_DATA)

# 왼쪽: 설문조사지 / 오른쪽: 분석 결과 및 그래프 배치
left_col, right_col = st.columns([1, 2.2])

with left_col:
    st.subheader("📋 설문조사지")
    st.caption("아래 문항에 응답하면 오른쪽 그래프에 실시간 반영됩니다.")
    with st.form("survey_sheet", clear_on_submit=True):
        grade = st.selectbox("1. 학년을 선택하세요.", ["1학년", "2학년", "3학년"], index=1)
        hours = st.slider("2. 평소 하루 평균 자기주도 공부 시간은?", 0.0, 10.0, 2.5, 0.5)
        score = st.number_input("3. 최근 정기고사 평균 점수는? (점)", 0, 100, 80, 1)
        method = st.radio(
            "4. 가장 주로 사용하는 공부 방법은?",
            ["문제집 풀이", "교과서·프린트 복습", "오답노트 정리", "인강·인터넷 강의"]
        )
        submit = st.form_submit_button("✅ 설문 응답 제출하기", use_container_width=True)

        if submit:
            now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
            new_row = pd.DataFrame([{
                "타임스탬프": now_str,
                "학년/반": grade,
                "하루 평균 공부 시간(시간)": float(hours),
                "시험 평균 점수(점)": float(score),
                "주된 공부 방법": method
            }])
            st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
            st.success("설문이 제출되어 그래프에 반영되었습니다!")

    with st.expander("📄 수집된 설문 응답 목록 보기", expanded=False):
        st.dataframe(st.session_state.df, use_container_width=True, hide_index=True)

with right_col:
    df = st.session_state.df.copy()
    x = df["하루 평균 공부 시간(시간)"].values
    y = df["시험 평균 점수(점)"].values
    corr = np.corrcoef(x, y)[0, 1]
    slope, intercept = np.polyfit(x, y, 1)

    st.subheader("📈 설문 데이터 회귀 분석 결과")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 응답자 수", f"{len(df)}명")
    c2.metric("평균 공부 시간", f"{np.mean(x):.2f}시간")
    c3.metric("상관계수 (r)", f"{corr:.3f}")
    c4.metric("회귀 기울기", f"+{slope:.2f}점/시간")

    # 회귀 산점도 그래프
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        name="설문 응답 데이터",
        marker=dict(size=10, color="#2E86C1", opacity=0.8),
        hovertemplate="공부 시간: %{x}시간<br>시험 점수: %{y}점<extra></extra>"
    ))
    x_line = np.linspace(max(0, x.min() - 0.3), x.max() + 0.3, 100)
    fig.add_trace(go.Scatter(
        x=x_line, y=slope * x_line + intercept,
        mode="lines", name="회귀 추세선",
        line=dict(color="#E74C3C", width=3, dash="dash")
    ))
    fig.update_layout(
        title="하루 평균 공부 시간과 시험 평균 점수의 회귀 분석",
        xaxis_title="하루 평균 공부 시간 (시간)",
        yaxis_title="시험 평균 점수 (점)",
        yaxis=dict(range=[50, 105]),
        height=360,
        margin=dict(t=40, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 하단 막대그래프 2개
    b1, b2 = st.columns(2)
    with b1:
        df["구간"] = pd.cut(df["하루 평균 공부 시간(시간)"], bins=[-0.1, 1.5, 3.0, 4.5, 24],
                          labels=["1.5h 이하", "1.5~3h", "3~4.5h", "4.5h 초과"])
        g1 = df.groupby("구간", observed=False)["시험 평균 점수(점)"].mean().reset_index()
        bar1 = go.Figure(go.Bar(x=g1["구간"].astype(str), y=g1["시험 평균 점수(점)"].round(1),
                                text=g1["시험 평균 점수(점)"].round(1), textposition="outside", marker_color="#5DADE2"))
        bar1.update_layout(title="공부 시간 구간별 평균 점수", yaxis=dict(range=[0, 110]), height=260, margin=dict(t=35, b=20))
        st.plotly_chart(bar1, use_container_width=True)

    with b2:
        g2 = df.groupby("주된 공부 방법")["시험 평균 점수(점)"].mean().reset_index()
        bar2 = go.Figure(go.Bar(x=g2["주된 공부 방법"], y=g2["시험 평균 점수(점)"].round(1),
                                text=g2["시험 평균 점수(점)"].round(1), textposition="outside", marker_color="#48C9B0"))
        bar2.update_layout(title="공부 방법별 평균 점수", yaxis=dict(range=[0, 110]), height=260, margin=dict(t=35, b=20))
        st.plotly_chart(bar2, use_container_width=True)
