import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="PISA 2022 숙제 시간과 수학 성취도 분석", layout="wide")

st.title("PISA 2022 한국 학생의 하루 숙제 시간과 수학 성취도 탐구")
st.markdown("PISA 2022 데이터를 활용해 한국 학생들의 하루 전체 숙제 시간과 수학 성취도(PV1~PV10)의 관계를 분석합니다.")

HOMEWORK_LABELS = {
    1: "숙제 없음 (0시간)",
    2: "30분 이하 (0.25시간)",
    3: "30분 초과 ~ 1시간 이하 (0.75시간)",
    4: "1시간 초과 ~ 2시간 이하 (1.5시간)",
    5: "2시간 초과 ~ 3시간 이하 (2.5시간)",
    6: "3시간 초과 ~ 4시간 이하 (3.5시간)",
    7: "4시간 초과 (4.5시간)"
}

HOMEWORK_HOURS = {
    1: 0.0,
    2: 0.25,
    3: 0.75,
    4: 1.5,
    5: 2.5,
    6: 3.5,
    7: 4.5
}

@st.cache_data
def read_any_file(file_or_path):
    name = file_or_path.lower() if isinstance(file_or_path, str) else file_or_path.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file_or_path, low_memory=False)
    elif name.endswith(".parquet"):
        return pd.read_parquet(file_or_path)
    elif name.endswith(".xlsx"):
        return pd.read_excel(file_or_path)
    elif name.endswith(".sav"):
        return pd.read_spss(file_or_path)
    return None

# 프로젝트 폴더 내 데이터 파일 자동 탐색
def find_local_data_files():
    patterns = ["**/*.csv", "**/*.sav", "**/*.parquet", "**/*.xlsx"]
    found = []
    for p in patterns:
        found.extend(glob.glob(p, recursive=True))
    # 결과 다운로드용 예시나 숨김 파일 제외
    return [f for f in found if not os.path.basename(f).startswith(".")]

local_files = find_local_data_files()

with st.sidebar:
    st.header("데이터 및 분석 설정")
    uploaded_file = st.file_uploader("다른 파일로 분석하려면 업로드", type=["csv", "sav", "parquet", "xlsx"])
    selected_local = None
    if local_files and uploaded_file is None:
        selected_local = st.selectbox("저장소 기본 데이터 파일", local_files, index=0)

# 데이터 로드 (업로드 파일 우선, 없으면 저장소 내 첫 번째 데이터 파일 자동 로드)
raw_df = None
data_source_name = ""

if uploaded_file is not None:
    raw_df = read_any_file(uploaded_file)
    data_source_name = uploaded_file.name
elif selected_local is not None:
    raw_df = read_any_file(selected_local)
    data_source_name = selected_local

if raw_df is None:
    st.error("저장소(GitHub)에서 데이터 파일을 찾지 못했습니다. GitHub 저장소에 PISA 데이터 파일(.csv 등)을 업로드해 주세요.")
    st.stop()

st.caption(f"불러온 데이터: `{data_source_name}`")

# 한국 데이터 필터링
if "CNT" in raw_df.columns:
    df = raw_df[raw_df["CNT"].astype(str).str.upper() == "KOR"].copy()
    if df.empty:
        df = raw_df.copy()
else:
    df = raw_df.copy()

# PV 컬럼 자동 탐색
pv_cols = [f"PV{i}MATH" for i in range(1, 11) if f"PV{i}MATH" in df.columns]
if not pv_cols:
    st.error("데이터에서 수학 성취도 컬럼(PV1MATH ~ PV10MATH)을 찾을 수 없습니다.")
    st.stop()

# 숙제 시간 컬럼 자동 선택 (가장 흔한 PISA 2022 변수명 우선)
hw_candidates = [col for col in ["ST296Q01JA", "ST296Q01HA", "HOMEWORK", "_homework_code"] if col in df.columns]
default_hw = hw_candidates[0] if hw_candidates else df.columns[0]
default_wt = "W_FSTUWT" if "W_FSTUWT" in df.columns else "가중치 사용 안 함"

with st.sidebar:
    hw_col = st.selectbox(
        "숙제 시간 변수 컬럼",
        options=df.columns.tolist(),
        index=df.columns.tolist().index(default_hw)
    )
    weight_options = ["가중치 사용 안 함"] + df.columns.tolist()
    weight_col = st.selectbox(
        "표본 가중치 컬럼",
        options=weight_options,
        index=weight_options.index(default_wt)
    )

# 전처리
data = df.copy()
data["_homework_code"] = pd.to_numeric(data[hw_col], errors="coerce")
data = data[data["_homework_code"].isin(HOMEWORK_HOURS.keys())].copy()
data["_homework_hours"] = data["_homework_code"].map(HOMEWORK_HOURS)

for pv in pv_cols:
    data[pv] = pd.to_numeric(data[pv], errors="coerce")
data = data.dropna(subset=["_homework_hours"] + pv_cols).copy()

if weight_col != "가중치 사용 안 함" and weight_col in data.columns:
    data["_weight"] = pd.to_numeric(data[weight_col], errors="coerce").fillna(1.0)
else:
    data["_weight"] = 1.0

if len(data) < 20:
    st.warning("유효한 분석 대상 데이터가 너무 적습니다. 왼쪽 사이드바에서 숙제 시간 변수 컬럼을 올바르게 선택해 주세요.")
    st.stop()

info = {"pv_cols": pv_cols}

# 통계 및 모델 계산
stat_rows = []
eval_rows = []

for pv in pv_cols:
    x = data["_homework_hours"].values
    y = data[pv].values
    w = data["_weight"].values

    w_sum = np.sum(w)
    x_mean = np.sum(w * x) / w_sum
    y_mean = np.sum(w * y) / w_sum
    cov_xy = np.sum(w * (x - x_mean) * (y - y_mean)) / w_sum
    var_x = np.sum(w * (x - x_mean) ** 2) / w_sum
    var_y = np.sum(w * (y - y_mean) ** 2) / w_sum

    slope = cov_xy / var_x if var_x > 0 else 0.0
    intercept = y_mean - slope * x_mean
    corr = cov_xy / np.sqrt(var_x * var_y) if (var_x > 0 and var_y > 0) else 0.0

    stat_rows.append({
        "PV": pv,
        "상관계수(r)": corr,
        "회귀 기울기": slope,
        "절편": intercept
    })

    if len(data) >= 50:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            x.reshape(-1, 1), y, w, test_size=0.2, random_state=42
        )
        base_pred = np.full_like(y_test, np.average(y_train, weights=w_train))
        reg = LinearRegression()
        reg.fit(X_train, y_train, sample_weight=w_train)
        reg_pred = reg.predict(X_test)

        eval_rows.append({
            "PV": pv,
            "기준모델 MAE": mean_absolute_error(y_test, base_pred, sample_weight=w_test),
            "회귀모델 MAE": mean_absolute_error(y_test, reg_pred, sample_weight=w_test),
            "기준모델 MSE": mean_squared_error(y_test, base_pred, sample_weight=w_test),
            "회귀모델 MSE": mean_squared_error(y_test, reg_pred, sample_weight=w_test),
        })

stat_df = pd.DataFrame(stat_rows)
eval_df = pd.DataFrame(eval_rows)

summary = {
    "corr": stat_df["상관계수(r)"].mean(),
    "slope": stat_df["회귀 기울기"].mean(),
    "intercept": stat_df["절편"].mean()
}

group_list = []
for code in sorted(HOMEWORK_LABELS.keys()):
    sub = data[data["_homework_code"] == code]
    if not sub.empty:
        pv_means = [np.average(sub[pv], weights=sub["_weight"]) for pv in pv_cols]
        group_list.append({
            "숙제 시간": HOMEWORK_LABELS[code],
            "학생 수": len(sub),
            "평균 수학 성취도": np.mean(pv_means)
        })
group_df = pd.DataFrame(group_list)

# 1. 데이터 미리보기
st.markdown("### 1. 분석 데이터 미리보기")
preview_cols = [c for c in ["CNT", "STDSCHID", "CNTSTUID", hw_col, "_homework_hours", "W_FSTUWT", "PV1MATH"] if c in data.columns]
st.dataframe(data[preview_cols].head(10), use_container_width=True, hide_index=True)

# 2. 핵심 분석 결과 요약
st.markdown("### 2. 핵심 분석 결과 요약")
m1, m2, m3 = st.columns(3)
m1.metric("분석 대상 학생 수", f"{len(data):,}명")
m2.metric("평균 상관계수 (r)", f"{summary['corr']:.3f}")
m3.metric("평균 회귀 기울기", f"{summary['slope']:.2f}점/시간")

if summary["corr"] > 0.1:
    corr_text = "양의 상관관계 경향"
elif summary["corr"] < -0.1:
    corr_text = "음의 상관관계 경향"
else:
    corr_text = "뚜렷하지 않은 선형 상관관계"

st.write(f"분석 결과, 두 변수 사이에는 **{corr_text}**가 나타났습니다.")
st.write(
    f"회귀 기울기는 약 **{summary['slope']:.2f}점/시간**입니다. "
    "이는 숙제 시간이 1시간 증가할 때 수학 성취도 추정값이 평균적으로 얼마나 달라지는지를 나타내는 "
    "회귀모형의 경향이며, 인과관계를 뜻하지는 않습니다."
)

with st.expander("PV별 분석 결과 확인"):
    st.dataframe(stat_df.round(4), use_container_width=True)

# 3. 회귀분석 그래프
st.markdown("### 3. 회귀분석 그래프")
st.caption(
    "점은 첫 번째 수학 성취도 추정값(PV1MATH)을 사용해 표시하고, "
    "회귀직선은 사용 가능한 모든 PV의 회귀선을 평균해 표시합니다. "
    "전체 점이 너무 많으면 그래프 표시만 일부 표본으로 줄이며, 계산에는 전체 자료를 사용합니다."
)

plot_df = data.copy()
if len(plot_df) > 6000:
    plot_df = plot_df.sample(n=6000, random_state=42)
pv1 = info["pv_cols"][0]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=plot_df["_homework_hours"],
    y=plot_df[pv1],
    mode="markers",
    name=f"학생 자료 ({pv1})",
    marker=dict(size=5, opacity=0.28),
    hovertemplate="하루 숙제 시간: %{x:.2f}시간<br>수학 성취도: %{y:.1f}<extra></extra>"
))

x_line = np.linspace(0, 4.7, 100)
y_line = summary["slope"] * x_line + summary["intercept"]
fig.add_trace(go.Scatter(
    x=x_line,
    y=y_line,
    mode="lines",
    name="평균 회귀직선",
    line=dict(width=3)
))
fig.update_layout(
    title="하루 전체 숙제 시간과 수학 성취도의 관계",
    xaxis_title="하루 전체 숙제 시간 (시간/일)",
    yaxis_title="수학 성취도 추정값",
    height=560,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="closest"
)
st.plotly_chart(fig, use_container_width=True)

# 4. 숙제 시간 구간별 통계
st.markdown("### 4. 숙제 시간 구간별 학생 수와 평균 성취도")
if not group_df.empty:
    show_group = group_df[["숙제 시간", "학생 수", "평균 수학 성취도"]].copy()
    show_group["평균 수학 성취도"] = show_group["평균 수학 성취도"].round(2)
    st.dataframe(show_group, use_container_width=True, hide_index=True)

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=group_df["숙제 시간"],
        y=group_df["평균 수학 성취도"],
        name="평균 수학 성취도",
        text=group_df["평균 수학 성취도"].round(1),
        textposition="outside"
    ))
    bar_fig.update_layout(
        title="하루 숙제 시간 구간별 평균 수학 성취도",
        xaxis_title="하루 전체 숙제 시간",
        yaxis_title="평균 수학 성취도 추정값",
        height=430
    )
    st.plotly_chart(bar_fig, use_container_width=True)

# 5. 기준모델과 회귀모델 비교
st.markdown("### 5. 기준모델과 회귀모델 비교")
if not eval_df.empty:
    mae_base = eval_df["기준모델 MAE"].mean()
    mae_reg = eval_df["회귀모델 MAE"].mean()
    mse_base = eval_df["기준모델 MSE"].mean()
    mse_reg = eval_df["회귀모델 MSE"].mean()

    e1, e2 = st.columns(2)
    e1.metric("기준모델 테스트 MAE", f"{mae_base:.2f}")
    e2.metric("회귀모델 테스트 MAE", f"{mae_reg:.2f}")

    compare = pd.DataFrame({
        "모델": ["평균값 기준모델", "선형 회귀모델"],
        "MAE": [mae_base, mae_reg],
        "MSE": [mse_base, mse_reg],
    })
    st.dataframe(compare.round(3), use_container_width=True, hide_index=True)
    if mae_reg < mae_base:
        st.success("테스트 자료에서 회귀모델의 MAE가 기준모델보다 작았습니다. 숙제 시간 변수를 이용한 예측이 기준모델보다 나았습니다.")
    elif mae_reg > mae_base:
        st.info("테스트 자료에서 회귀모델의 MAE가 기준모델보다 컸습니다. 숙제 시간만으로 성취도를 예측하는 데 한계가 있음을 보여줍니다.")
    else:
        st.info("두 모델의 테스트 MAE가 같습니다.")
    with st.expander("PV별 모델 비교"):
        st.dataframe(eval_df.round(3), use_container_width=True)
else:
    st.info("테스트 자료가 충분하지 않아 모델 비교를 계산하지 못했습니다.")

# 6. 주의할 점
st.markdown("### 6. 탐구 결과 해석 시 주의할 점")
st.markdown(
    """
- 이 분석은 숙제 시간과 수학 성취도의 **관계**를 확인하는 것이며, 숙제 시간이 성취도를 직접 높인다는 인과관계를 증명하지 않습니다.
- 가정의 사회경제적 배경, 사교육, 수업의 질, 학습 태도 등 다른 요인이 결과에 영향을 줄 수 있습니다.
- PISA는 만 15세 학생을 대상으로 한 국제평가이며 일반적인 학교 수학 시험 점수와는 다릅니다.
- PISA의 정확한 모집단 추정에는 표본 가중치와 반복 가중치를 이용한 표준오차 계산이 필요합니다. 이 앱은 표본 가중치를 반영한 탐색적 분석이며, 반복 가중치를 이용한 공식 표준오차 계산까지 수행하지는 않습니다.
- OECD가 발표한 숙제 시간 관련 16점 차이는 OECD 국가 평균의 별도 분석 결과입니다. 이 앱의 한국 학생 분석 결과와 구분해서 해석해야 합니다.
    """
)

# 7. 결과 내려받기
st.markdown("### 7. 결과 내려받기")
download_df = data.copy()
download_df["하루 전체 숙제 시간(시간)"] = download_df["_homework_hours"]
download_df["숙제 시간 응답"] = download_df["_homework_code"].map(HOMEWORK_LABELS)
csv_bytes = download_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button(
    "전처리된 한국 학생 분석 자료 CSV 다운로드",
    data=csv_bytes,
    file_name="pisa2022_korea_homework_math_clean.csv",
    mime="text/csv"
)
