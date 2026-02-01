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
# كشف وضع المتصفح تلقائياً (نهاري أو داكن)
# -------------------------------------------------
dark_mode_js = """
<script>
const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
document.body.setAttribute('data-dark', isDark);
</script>
"""
st.components.v1.html(dark_mode_js, height=0)

# -------------------------------------------------
# اختيار الألوان حسب المتصفح
# -------------------------------------------------
# نستخدم session_state لتخزين الوضع الداكن
if 'dark_mode' not in st.session_state:
    # الوضع الافتراضي: نستخدم الـ JS attribute لاحقاً للتحديث
    st.session_state.dark_mode = False

# -------------------------------------------------
# ألوان حسب الوضع
# -------------------------------------------------
def get_colors():
    if st.session_state.dark_mode:
        return {
            "bg": "#0e1117",
            "text": "#ffffff",
            "grid": "rgba(255,255,255,0.1)",
            "legend": "#E0E0E0",
            "hover": "#262730"
        }
    else:
        return {
            "bg": "#ffffff",
            "text": "#000000",
            "grid": "rgba(0,0,0,0.1)",
            "legend": "#333333",
            "hover": "#f0f2f6"
        }

colors = get_colors()

# -------------------------------------------------
# CSS لتثبيت الشريط على اليسار + ألوان ديناميكية
# -------------------------------------------------
st.markdown(f"""
<style>
section[data-testid="stSidebar"] {{
    left: 0 !important;
    right: auto !important;
    direction: ltr !important;
}}
section[data-testid="stSidebar"][style*="right: 0px"] {{
    display: none !important;
}}

.stApp {{ background-color: {colors['bg']}; color: {colors['text']}; }}
[data-testid="stAppViewContainer"] {{ direction: rtl; text-align: right; }}

div[data-testid="stMetric"] {{
    background-color: rgba(128, 128, 128, 0.1) !important;
    border: 1px solid rgba(128, 128, 128, 0.2) !important;
    padding: 20px !important; border-radius: 15px !important;
    text-align: center !important;
}}
[data-testid="stMetricLabel"] div p {{
    color: {colors['text']} !important;
    font-weight: 900 !important;
    font-size: 18px !important;
    opacity:1;
}}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ background-color: transparent !important; }}

.header-style {{ font-size: clamp(24px, 5vw, 38px); font-weight: 900; color: #0077b6; margin-bottom: 5px; }}
.region-style {{ font-size: 20px; color: {colors['text']}; margin-bottom: 30px; font-weight: 700; opacity: 0.8; }}
.sub-header {{ font-size: 24px; font-weight: 700; color: {colors['text']}; margin-bottom: 15px; margin-top: 15px; }}
.advice-card {{ background-color: rgba(0, 119, 182, 0.08); border-right: 6px solid #0077b6; padding: 25px; border-radius: 12px; margin-top: 25px; }}

.sidebar-btn {{ display: block !important; width: 100%; padding: 12px; margin-bottom: 10px; text-align: center; border-radius: 8px; text-decoration: none !important; font-weight: bold; color: white !important; }}
.wa-btn {{ background-color: #25D366; }} .li-btn {{ background-color: #0077B5; }}

.stDownloadButton button {{
    width:100%;
    background-color:#0077b6 !important;
    color:white !important;
    border-radius:8px !important;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# تحميل النماذج
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
# إعدادات السايدبار
# -------------------------------------------------
with st.sidebar:
    st.title("معاذ عثمان")
    st.session_state.dark_mode = st.toggle("🌙 الوضع الليلي", value=st.session_state.dark_mode)
    st.info("نظام التنبؤ المتقدم (Prophet Engine)")
    st.markdown(f'<a href="https://wa.me/249919640534" class="sidebar-btn wa-btn">💬 واتساب</a>', unsafe_allow_html=True)
    st.markdown(f'<a href="https://www.linkedin.com/in/moaazos/" class="sidebar-btn li-btn">🔗 لينكد إن</a>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("⚙️ الإعدادات")
    selected_region = st.selectbox("المنطقة الجغرافية", ["الكل", "Central", "South", "East", "West"])
    all_cats = ["الأثاث", "الأدوات المكتبية", "التكنولوجيا"]
    selected_cat = st.selectbox("القطاع الرئيسي", all_cats)
    compare_cats = st.multiselect("قطاعات للمقارنة", [c for c in all_cats if c != selected_cat])
    forecast_months = st.number_input("أشهر التنبؤ", min_value=1, max_value=36, value=12)

# -------------------------------------------------
# دالة استخراج التنبؤ التفصيلي
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
st.markdown(f'<div class="header-style">Smart Sales Predictor (متنبئ المبيعات الذكي): {selected_cat}</div>', unsafe_allow_html=True)
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

    # 2. الرسم البياني الأساسي
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
        template="none", height=450, hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=colors['text']),
        legend=dict(font=dict(color=colors['legend'])),
        xaxis=dict(gridcolor=colors['grid'], tickfont=dict(color=colors['text'])),
        yaxis=dict(gridcolor=colors['grid'], tickfont=dict(color=colors['text'])),
        hoverlabel=dict(bgcolor=colors['hover'], font_color=colors['text'])
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. جدول البيانات وزر التحميل
    st.markdown('<div class="sub-header">📋 جدول البيانات التنبؤية الكامل</div>', unsafe_allow_html=True)
    display_df = df_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    display_df['ds'] = display_df['ds'].dt.strftime('%Y-%m-%d')
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
                template="none", height=450,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=colors['text']),
                xaxis=dict(gridcolor=colors['grid'], tickfont=dict(color=colors['text']), automargin=True),
                yaxis=dict(gridcolor=colors['grid'], tickfont=dict(color=colors['text']), automargin=True),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_multi, use_container_width=True)
        with col_c:
            fig_pie = px.pie(pd.DataFrame(pie_data), values='المبيعات', names='القطاع', hole=0.6, template="none",
                             color_discrete_sequence=['#0077b6', '#00b4d8', '#90e0ef', '#caf0f8'])
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=450, margin=dict(t=80, b=50, l=10, r=10),
                                  paper_bgcolor='rgba(0,0,0,0)', font=dict(color=colors['text'], size=13), showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

    # 5. التوصيات
    st.markdown('<div class="advice-card">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">💡 توصيات ذكاء الأعمال</div>', unsafe_allow_html=True)
    if growth > 10:
        advice_text = f"🚀 نمو بنسبة {growth:.1f}% متوقع."
    elif growth < 0:
        advice_text = f"⚠️ تراجع بنسبة {growth:.1f}% متوقع. يُنصح بمراجعة استراتيجية المبيعات."
    else:
        advice_text = "📊 استقرار نسبي."
    st.write(advice_text)
    st.write(f"🔍 متوسط تذبذب التوقعات: ${confidence_range:,.0f}.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 6. مساهمة المناطق
    if selected_region == "الكل":
        st.markdown('<div class="sub-header">تحليل مساهمة المناطق</div>', unsafe_allow_html=True)
        contrib_df = pd.concat([d.tail(forecast_months) for d in regional_list])
        fig_area = px.area(contrib_df, x="ds", y="yhat", color="region", template="none",
                           color_discrete_sequence=px.colors.sequential.Blues_r)
        fig_area.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=colors['text']))
        st.plotly_chart(fig_area, use_container_width=True)

    # 7. المكونات
    if base_model:
        st.markdown('<div class="sub-header">تحليل المكونات (الموسمية والاتجاه)</div>', unsafe_allow_html=True)
        try:
            fig_comp = plot_components_plotly(base_model, full_forecast)
            fig_comp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=colors['text']))
            fig_comp.update_xaxes(tickfont=dict(color=colors['text']), gridcolor=colors['grid'])
            fig_comp.update_yaxes(tickfont=dict(color=colors['text']), gridcolor=colors['grid'])
            st.plotly_chart(fig_comp, use_container_width=True)
        except:
            st.info("تحليل المكونات متاح للمناطق الفردية.")

st.markdown(f"<hr><div style='text-align: center; opacity: 0.6; color: {colors['text']};'>تطوير: معاذ عثمان | 2026</div>", unsafe_allow_html=True)
