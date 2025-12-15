import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pytz
from io import StringIO
from fpdf import FPDF
import base64

# --- CONFIGURATION ---
st.set_page_config(page_title="PowerGrid AI", page_icon="⚡", layout="wide")

API_URL = "http://localhost:8000"

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #0068c9; color: white; border-radius: 8px; height: 3em; font-weight: bold; }
    .metric-box { border: 1px solid #e0e0e0; padding: 20px; border-radius: 10px; background: white; text-align: center; }
    .live-badge { background-color: #ff4b4b; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold; font-size: 0.8em; animation: pulse 2s infinite;}
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    </style>
""", unsafe_allow_html=True)

# --- HELPER: LIVE WEATHER API ---
def get_live_weather(lat=34.07, lon=72.68):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(url, timeout=2)
        return response.json()['current_weather']['temperature']
    except:
        return None

# --- HELPER: PDF GENERATOR ---
def create_pdf(demand, risk, hour, temp, cost):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="PowerGrid AI - Status Report", ln=1, align='C')
    
    pdf.set_font("Arial", size=12)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pdf.cell(200, 10, txt=f"Generated: {timestamp}", ln=1, align='C')
    pdf.line(10, 30, 200, 30)
    pdf.ln(20)
    
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt=f"Grid Load: {demand:.2f} kW", ln=1)
    pdf.cell(200, 10, txt=f"Risk Status: {risk}", ln=1)
    pdf.cell(200, 10, txt=f"Estimated Cost: Rs. {cost:.2f}", ln=1)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Conditions: Temp {temp}C at Hour {hour}", ln=1)
    
    return pdf.output(dest='S').encode('latin-1')

# --- SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚡ PowerGrid AI")
    st.markdown("---")
    page = st.radio("Navigation", ["Real-time Dashboard (Manual)", "Live Grid Monitor (API)", "Batch Analytics"])
    st.markdown("---")
    
    if st.button("Check Backend Status"):
        try:
            r = requests.get(f"{API_URL}/docs", timeout=1)
            if r.status_code == 200: st.success("Online 🟢")
            else: st.error(f"Error: {r.status_code}")
        except: st.error("Offline 🔴")

# --- PAGE 1: MANUAL DASHBOARD ---
if page == "Real-time Dashboard (Manual)":
    st.title("🎛️ Manual Simulator")
    
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            days = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
            day_name = st.selectbox("Day", list(days.keys()), index=2)
            day_int = days[day_name]
        with c2:
            hour = st.slider("Hour", 0, 23, 12)
        with c3:
            temp = st.number_input("Temp (°C)", 10.0, 45.0, 25.0, 1.0)
        with c4:
            volt = st.number_input("Voltage (V)", 200.0, 250.0, 230.0, 1.0, help=">236: Low Risk | 228-236: Moderate | <228: High")

    if st.button("Simulate"):
        payload = {"hour": int(hour), "temperature": float(temp), "voltage": float(volt), "dayofweek": int(day_int)}
        try:
            r_dem = requests.post(f"{API_URL}/predict-demand", json=payload)
            r_risk = requests.post(f"{API_URL}/peak-hour", json=payload)
            
            if r_dem.status_code == 200:
                d_val = r_dem.json()['predicted_demand']
                r_val = r_risk.json()['risk']
                
                # Logic
                is_peak = 17 <= hour <= 22
                rate = 25.0 if is_peak else 15.0
                cost = d_val * rate
                st.session_state.history.append({"Hour": hour, "Demand": d_val, "Cost": cost})

                color = "#28a745"
                if "Moderate" in r_val: color = "#ffc107"
                if "High" in r_val: color = "#dc3545"

                # UI
                k1, k2, k3 = st.columns(3)
                k1.metric("Predicted Demand", f"{d_val:.2f} kW")
                k2.metric("Est. Hourly Cost", f"Rs. {cost:.2f}", f"{rate} Rs/Unit ({'Peak' if is_peak else 'Off'})")
                k3.markdown(f"<div style='text-align:center; padding:10px; background:{color}; color:white; border-radius:5px;'><h3>{r_val}</h3></div>", unsafe_allow_html=True)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number", value = d_val,
                    gauge = {'axis': {'range': [None, 8]}, 'bar': {'color': color}}
                ))
                fig.update_layout(height=250, margin=dict(t=30, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("ℹ️ Why this prediction?"):
                    reasons = []
                    if volt < 228: reasons.append("CRITICAL: Low Voltage (<228V) indicates grid stress.")
                    if temp > 30: reasons.append("High Temperature (>30°C) is driving cooling load.")
                    if is_peak: reasons.append("Time is within Peak Hours (5 PM - 10 PM).")
                    if not reasons: st.write("✅ All parameters are within normal stable ranges.")
                    else: 
                        for r in reasons: st.write(f"- {r}")

                col_pdf, _ = st.columns([1,4])
                with col_pdf:
                    pdf_bytes = create_pdf(d_val, r_val, hour, temp, cost)
                    st.download_button("📄 Download Report", data=pdf_bytes, file_name="grid_report.pdf", mime="application/pdf")

                if len(st.session_state.history) > 0:
                    st.markdown("### 🕒 Session Trend")
                    hist_df = pd.DataFrame(st.session_state.history)
                    st.line_chart(hist_df.set_index("Hour")["Demand"])

        except Exception as e:
            st.error(f"Error: {e}")

# --- PAGE 2: LIVE GRID MONITOR ---
elif page == "Live Grid Monitor (API)":
    st.markdown('## 🌍 Live Grid Monitor <span class="live-badge">LIVE DATA</span>', unsafe_allow_html=True)
    
    if st.button("🔄 Refresh"): st.cache_data.clear()

    try: tz = pytz.timezone('Asia/Karachi')
    except: tz = pytz.utc
        
    now = datetime.now(tz)
    current_hour = now.hour
    current_minute = now.minute
    current_day_int = now.weekday()
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    time_str = f"{current_hour:02d}:{current_minute:02d}"
    
    with st.spinner("Fetching Satellite Data..."):
        current_temp = get_live_weather(34.07, 72.68)
    
    if current_temp is None: current_temp = 25.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Time", time_str, day_names[current_day_int])
    col2.metric("Temp", f"{current_temp} °C", "Topi, PK")
    with col3: live_volt = st.slider("Sensor (V)", 200, 250, 230)
    col4.metric("Voltage", f"{live_volt} V")

    st.markdown("---")

    if st.button("RUN LIVE ANALYSIS", type="primary"):
        payload = {"hour": current_hour, "temperature": float(current_temp), "voltage": float(live_volt), "dayofweek": current_day_int}
        try:
            r1 = requests.post(f"{API_URL}/predict-demand", json=payload)
            r2 = requests.post(f"{API_URL}/peak-hour", json=payload)
            
            if r1.status_code == 200:
                dem = r1.json()['predicted_demand']
                risk = r2.json()['risk']
                
                is_peak = 17 <= current_hour <= 22
                rate = 25.0 if is_peak else 15.0
                cost = dem * rate

                c_main = "#28a745"
                if "Moderate" in risk: c_main = "#ffc107"
                if "High" in risk: c_main = "#dc3545"
                
                m1, m2 = st.columns(2)
                with m1:
                    st.info(f"⚡ Load: **{dem:.2f} kW** | Cost: **Rs. {cost:.2f}**")
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number", value = dem,
                        gauge = {'axis': {'range': [None, 8]}, 'bar': {'color': c_main}}
                    ))
                    fig.update_layout(height=250, margin=dict(t=10,b=10,l=10,r=10))
                    st.plotly_chart(fig, use_container_width=True)
                
                with m2:
                    st.warning(f"🛡️ Status: {risk}")
                    with st.expander("Automated Action Plan", expanded=True):
                        if "High" in risk: st.error("1. Start Backup Gen\n2. Shed Load Zone A")
                        elif "Moderate" in risk: st.warning("1. Monitor Voltage\n2. Alert Team")
                        else: st.success("System Normal")
                    
                    pdf_bytes = create_pdf(dem, risk, current_hour, current_temp, cost)
                    st.download_button("📄 Save Log", data=pdf_bytes, file_name=f"live_log_{current_hour}h.pdf")

        except Exception as e: st.error(f"Error: {e}")

# --- PAGE 3: BATCH ANALYTICS ---
elif page == "Batch Analytics":
    st.title("📂 Batch Processing")
    up_file = st.file_uploader("Upload CSV", type=["csv", "txt"])
    
    if up_file and st.button("Process Batch File"):
        files = {"file": (up_file.name, up_file.getvalue(), "text/csv")}
        try:
            res = requests.post(f"{API_URL}/upload-data", files=files).json()
            preds = res.get("predictions", []) if isinstance(res, dict) else res
            
            up_file.seek(0)
            df = pd.read_csv(up_file).dropna()
            
            if len(preds) == len(df):
                df['Prediction'] = preds
                st.success(f"Processed {len(df)} rows successfully!")
                st.dataframe(df)
                if 'hour' in df.columns: 
                    st.markdown("### 📈 Batch Trend")
                    st.line_chart(df.set_index('hour')['Prediction'])
            else:
                st.error("Row mismatch. Check CSV for empty lines.")
        except Exception as e: st.error(f"Error: {e}")