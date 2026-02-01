import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from prophet.plot import plot_components_plotly
import os
import io

# -------------------------------------------------
# إعدادات الصفحة
# -------------------------------------------------
st.set_page_config(page_title="Prophet Sales Intelligence | معاذ عثمان", layout="wide")

# -------------------------------------------------
# CSS:  الواجهة العلوية
# -------------------------------------------------
st.markdown(f"""
<style>
/* إخفاء السايدبار تماماً */
[data-testid="stSidebar"] {{
    display: none !important;
}}

/* ضبط مساحة الصفحة لتستغل كامل العرض */
[data-testid="stAppViewContainer"] {{
    direction: rtl;
    text-align: right;
    padding-top: 2rem;
}}

div[data-testid="stMetric"] {{
    background-color: rgba(128, 128, 128, 0.1) !important;
    border: 1px solid rgba(128, 128, 128, 0.2) !important;
    padding: 20px !important; border-radius: 15px !important;
    text-align: center !important;
}}

[data-testid="stMetricLabel"] div p {{
    font-weight: 900 !important;
    font-size: 18px !important;
    opacity: 1 !important;
}}

.header-style {{ font-size: clamp(24px, 5vw, 38px); font-weight: 900; color: #0077b6; margin-bottom: 5px; }}
.region-style {{ font-size: 20px; margin-bottom: 30px; font-weight: 700; opacity: 0.8; }}
.sub-header {{ font-size: 24px; font-weight: 700; margin-bottom: 15px; margin-top: 15px; }}
.advice-card {{ background-color: rgba(0, 119, 182, 0.08); border-right: 6px solid #0077b6; padding: 25px; border-radius: 12px; margin-top: 25px; }}

/* تنسيق أزرار التواصل العلوية */
.top-btn-container {{ display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
.top-btn {{ padding: 10px 20px; border-radius: 8px; text-decoration: none !important; font-weight: bold; color: white !important; display: inline-block; }}
.wa-btn {{ background-color: #25D366; }} .li-btn {{ background-color: #0077B5; }}

.stDownloadButton button {{
    width: 100%;
    background-color: #0077b6 !important;
    color: white !important;
    border-radius: 8px !important;
}}

/* تحسين شكل المدخلات في الأعلى */
.stSelectbox, .stNumberInput, .stMultiSelect {{
    border-radius: 10px;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# دالة تحميل النماذج
# -------------------------------------------------
@st.cache_resource
def load_prophet_engine():
    categories = {"الأثاث": "furniture", "الأدوات المكتبية": "office_supplies", "التكنولوجيا": "technology"}
    regions = ["Central", "South", "East", "West"]
    loaded_models = {}
    for ar_cat, en_cat in categories.items():
        for reg in regions:
            p_name = f"prophet_{en_cat}_{reg.lower()}.pkl"
            s_name = f"sarima_{en_cat}_{reg.lower()}.pkl"
            file_to_load = p_name if os.path.exists(p_name) else s_name if os.path.exists(s_name) else None
            if file_to_load:
                with open(file_to_load, "rb") as f:
                    loaded_models[f"{ar_cat}_{reg}"] = pickle.load(f)
    return loaded_models

models = load_prophet_engine()

# -------------------------------------------------
# منطقة إعدادات علوية (Top Navigation)
# -------------------------------------------------
st.markdown('<div class="header-style">معاذ عثمان | نظام التنبؤ للمبيعات</div>', unsafe_allow_html=True)

# أزرار التواصل في الأعلى
st.markdown(f"""
<div class="top-btn-container">
    <a href="https://wa.me/249919640534" class="top-btn wa-btn">💬 واتساب</a>
    <a href="https://www.linkedin.com/in/moaazos/" class="top-btn li-btn">🔗 لينكد إن</a>
</div>
""", unsafe_allow_html=True)

# منطقة الإعدادات
with st.expander("⚙️ إعدادات التحكم والفلترة", expanded=True):
    col_set1, col_set2, col_set3, col_set4 = st.columns(4)
    with col_set1:
        selected_region = st.selectbox("المنطقة الجغرافية", ["الكل", "Central", "South", "East", "West"])
    with col_set2:
        all_cats = ["الأثاث", "الأدوات المكتبية", "التكنولوجيا"]
        selected_cat = st.selectbox("القطاع الرئيسي", all_cats)
    with col_set3:
        compare_cats = st.multiselect("قطاعات للمقارنة", [c for c in all_cats if c != selected_cat])
    with col_set4:
        forecast_months = st.number_input("أشهر التنبؤ", min_value=1, max_value=36, value=12)

st.markdown("---")

# -------------------------------------------------
# دالة استخراج التنبؤ
# -------------------------------------------------
def get_detailed_forecast(cat, region, months):
    regs = ["Central", "South", "East", "West"] if region == "الكل" else [region]
    regional_data = []
    combined_df = None
    last_model = None
    for r in regs:
        key = f"{cat}_{r}"
        if key in models:
            m = models[key]
            last_model = m
            future = m.make_future_dataframe(periods=months, freq='MS')
            res = m.predict(future)
            df_res = res.copy()
            df_res['region'] = r
            regional_data.append(df_res)
            if combined_df is None:
                combined_df = df_res.copy()
            else:
                for col in ['yhat', 'yhat_lower', 'yhat_upper', 'trend']:
                    if col in combined_df.columns:
                        combined_df[col] += df_res[col]
    return combined_df, regional_data, last_model

# -------------------------------------------------
# العرض الرئيسي
# -------------------------------------------------
st.markdown(f'<div class="header-style">Sales Predictor: {selected_cat}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="region-style">📍 النطاق الجغرافي: {selected_region}</div>', unsafe_allow_html=True)

full_forecast, regional_list, base_model = get_detailed_forecast(selected_cat, selected_region, forecast_months)

if full_forecast is not None:
    df_forecast = full_forecast.tail(forecast_months).copy()
    
    # 1. المقاييس
    c1, c2, c3 = st.columns(3)
    total_sales = df_forecast['yhat'].sum()
    growth = ((df_forecast['yhat'].iloc[-1] - df_forecast['yhat'].iloc[0]) / df_forecast['yhat'].iloc[0]) * 100
    confidence_range = (df_forecast['yhat_upper'] - df_forecast['yhat_lower']).mean()
    
    c1.metric("إجمالي مبيعات الفترة", f"${total_sales:,.0f}")
    c2.metric("معدل النمو المتوقع", f"{growth:+.1f}%")
    c3.metric("نطاق اليقين (95%)", f"${confidence_range:,.0f}")

    # 2. الرسم البياني (تعديل ليتكيف تلقائياً)
    st.markdown('<div class="sub-header">المسار التنبؤي للمبيعات</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([df_forecast['ds'], df_forecast['ds'][::-1]]),
        y=pd.concat([df_forecast['yhat_upper'], df_forecast['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(0, 119, 182, 0.15)',
        line=dict(color='rgba(255,255,255,0)'), name='نطاق اليقين'
    ))
    fig.add_trace(go.Scatter(
        x=df_forecast['ds'], y=df_forecast['yhat'],
        mode='lines+markers', line=dict(color='#0077b6', width=3),
        name='التوقع الرئيسي'
    ))
    
    fig.update_layout(
        height=450, hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(automargin=True), yaxis=dict(automargin=True)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. جدول البيانات المعدل
    st.markdown('<div class="sub-header">📋 جدول البيانات التنبؤية الكامل</div>', unsafe_allow_html=True)
    display_df = df_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    display_df['ds'] = display_df['ds'].dt.strftime('%Y-%m-%d')
    display_df.columns = ['التاريخ', 'المبيعات المتوقعة', 'الحد الأدنى المتوقع', 'الحد الأعلى المتوقع']
    
    col_table, col_download = st.columns([4, 1])
    with col_table:
        st.dataframe(display_df, use_container_width=True)
    with col_download:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            display_df.to_excel(writer, index=False, sheet_name='Forecast')
        st.download_button(
            label="📥 تحميل التقرير (Excel)", 
            data=output.getvalue(), 
            file_name=f"sales_forecast_{selected_cat}.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # 4. مقارنة القطاعات
    if compare_cats:
        st.markdown("---")
        st.markdown('<div class="sub-header">مقارنة القطاعات</div>', unsafe_allow_html=True)
        col_p, col_c = st.columns([2, 1])
        pie_data = [{'القطاع': selected_cat, 'المبيعات': total_sales}]
        with col_p:
            fig_multi = go.Figure()
            fig_multi.add_trace(go.Scatter(x=df_forecast['ds'], y=df_forecast['yhat'], name=selected_cat, line=dict(color='#0077b6', width=3)))
            for cat in compare_cats:
                comp_f, _, _ = get_detailed_forecast(cat, selected_region, forecast_months)
                if comp_f is not None:
                    comp_tail = comp_f.tail(forecast_months)
                    fig_multi.add_trace(go.Scatter(x=comp_tail['ds'], y=comp_tail['yhat'], name=cat, line=dict(dash='dot')))
                    pie_data.append({'القطاع': cat, 'المبيعات': comp_tail['yhat'].sum()})
            fig_multi.update_layout(
                height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_multi, use_container_width=True)
        with col_c:
            fig_pie = px.pie(pd.DataFrame(pie_data), values='المبيعات', names='القطاع', hole=0.6, color_discrete_sequence=['#0077b6', '#00b4d8', '#90e0ef', '#caf0f8'])
            fig_pie.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

    # 5. التوصيات
    st.markdown('<div class="advice-card">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">💡 توصيات ذكاء الأعمال</div>', unsafe_allow_html=True)
    advice_text = f"🚀 نمو بنسبة {growth:.1f}% متوقع." if growth > 10 else f"⚠️ تراجع بنسبة {growth:.1f}% متوقع." if growth < 0 else "📊 استقرار نسبي."
    st.write(advice_text)
    st.write(f"🔍 متوسط تذبذب التوقعات: ${confidence_range:,.0f}.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 6. مساهمة المناطق)
    if selected_region == "الكل":
        st.markdown('<div class="sub-header">تحليل مساهمة المناطق</div>', unsafe_allow_html=True)
        contrib_df = pd.concat([d.tail(forecast_months) for d in regional_list])
        fig_area = px.area(contrib_df, x="ds", y="yhat", color="region", color_discrete_sequence=px.colors.sequential.Blues_r)
        fig_area.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_area, use_container_width=True)

    # 7. المكونات)
    if base_model:
        st.markdown('<div class="sub-header">تحليل المكونات (الموسمية والاتجاه)</div>', unsafe_allow_html=True)
        try:
            fig_comp = plot_components_plotly(base_model, full_forecast)
            fig_comp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_comp, use_container_width=True)
        except:
            st.info("تحليل المكونات متاح للمناطق الفردية.")

st.markdown(f"<hr><div style='text-align: center; opacity: 0.6;'>تطوير: معاذ عثمان | 2026</div>", unsafe_allow_html=True)
