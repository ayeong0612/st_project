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

st.markdown("### 4. 회귀분석 그래프")
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

st.markdown("### 5. 숙제 시간 구간별 학생 수와 평균 성취도")
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

st.markdown("### 6. 기준모델과 회귀모델 비교")
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

st.markdown("### 7. 탐구 결과 해석 시 주의할 점")
st.markdown(
    """
    - 이 분석은 숙제 시간과 수학 성취도의 **관계**를 확인하는 것이며, 숙제 시간이 성취도를 직접 높인다는 인과관계를 증명하지 않습니다.
    - 가정의 사회경제적 배경, 사교육, 수업의 질, 학습 태도 등 다른 요인이 결과에 영향을 줄 수 있습니다.
    - PISA는 만 15세 학생을 대상으로 한 국제평가이며 일반적인 학교 수학 시험 점수와는 다릅니다.
    - PISA의 정확한 모집단 추정에는 표본 가중치와 반복 가중치를 이용한 표준오차 계산이 필요합니다. 이 앱은 표본 가중치를 반영한 탐색적 분석이며, 반복 가중치를 이용한 공식 표준오차 계산까지 수행하지는 않습니다.
    - OECD가 발표한 숙제 시간 관련 16점 차이는 OECD 국가 평균의 별도 분석 결과입니다. 이 앱의 한국 학생 분석 결과와 구분해서 해석해야 합니다.
    """
)

st.markdown("### 8. 결과 내려받기")
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
