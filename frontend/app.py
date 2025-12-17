import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import pytz
import requests
from io import BytesIO
from fpdf import FPDF
import joblib
from pathlib import Path

# --- CONFIGURATION ---
st.set_page_config(page_title="PowerGrid AI", page_icon="⚡", layout="wide")

# --- CSS STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-box { border: 1px solid #e0e0e0; padding: 20px; border-radius: 10px; background: white; text-align: center; }
    .live-badge { background-color: #ff4b4b; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold; font-size: 0.8em; animation: pulse 2s infinite;}
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    </style>
""", unsafe_allow_html=True)

# --- HELPER: LOAD MODELS (Unified Architecture) ---
@st.cache_resource
def load_models():
    # Paths to your models
    reg_path = Path("models/regression.pkl")
    clf_path = Path("models/classifier.pkl")
    
    loaded_models = {}
    
    if reg_path.exists():
        loaded_models['regression'] = joblib.load(reg_path)
    if clf_path.exists():
        loaded_models['classifier'] = joblib.load(clf_path)
        
    return loaded_models

# Load models once
models = load_models()

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
    
    # Status Check
    if 'regression' in models and 'classifier' in models:
        st.success("Models Loaded 🟢")
    else:
        st.error("Models Missing 🔴")

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
        if 'regression' in models and 'classifier' in models:
            try:
                # Prepare Input
                X = [[hour, temp, volt, day_int]]
                
                # Predict
                d_val = models['regression'].predict(X)[0]
                risk_idx = models['classifier'].predict(X)[0]
                
                # Map Risk
                labels = {0: "Low Risk", 1: "High Load Shedding Risk"} # Binary Mapping
                r_val = labels.get(int(risk_idx), "Unknown")
                
                # Calculations
                is_peak = 17 <= hour <= 22
                rate = 25.0 if is_peak else 15.0
                cost = d_val * rate
                
                # Save to History
                st.session_state.history.append({"Hour": hour, "Demand": d_val, "Cost": cost})

                # Color Logic
                color = "#28a745"
                if "High" in r_val: color = "#dc3545"

                # UI
                k1, k2, k3 = st.columns(3)
                k1.metric("Predicted Demand", f"{d_val:.2f} kW")
                k2.metric("Est. Hourly Cost", f"Rs. {cost:.2f}", f"{rate} Rs/Unit ({'Peak' if is_peak else 'Off'})")
                k3.markdown(f"<div style='text-align:center; padding:10px; background:{color}; color:white; border-radius:5px;'><h3>{r_val}</h3></div>", unsafe_allow_html=True)
                
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number", value = d_val,
                    gauge = {'axis': {'range': [None, 8]}, 'bar': {'color': color}}
                ))
                fig.update_layout(height=250, margin=dict(t=30, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

                # Explainer
                with st.expander("ℹ️ Why this prediction?"):
                    reasons = []
                    if volt < 228: reasons.append("CRITICAL: Low Voltage (<228V) indicates grid stress.")
                    if temp > 30: reasons.append("High Temperature (>30°C) is driving cooling load.")
                    if is_peak: reasons.append("Time is within Peak Hours (5 PM - 10 PM).")
                    if not reasons: st.write("✅ All parameters are within normal stable ranges.")
                    else: 
                        for r in reasons: st.write(f"- {r}")

                # PDF Download
                col_pdf, _ = st.columns([1,4])
                with col_pdf:
                    pdf_bytes = create_pdf(d_val, r_val, hour, temp, cost)
                    st.download_button("📄 Download Report", data=pdf_bytes, file_name="grid_report.pdf", mime="application/pdf")

                # Trend Chart
                if len(st.session_state.history) > 0:
                    st.markdown("### 🕒 Session Trend")
                    hist_df = pd.DataFrame(st.session_state.history)
                    st.line_chart(hist_df.set_index("Hour")["Demand"])

            except Exception as e:
                st.error(f"Prediction Error: {e}")
        else:
            st.error("Models failed to load.")

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
        if 'regression' in models and 'classifier' in models:
            # Prepare Input
            X = [[current_hour, current_temp, live_volt, current_day_int]]
            
            # Predict
            dem = models['regression'].predict(X)[0]
            risk_idx = models['classifier'].predict(X)[0]
            
            labels = {0: "Low Risk", 1: "High Load Shedding Risk"}
            risk = labels.get(int(risk_idx), "Unknown")
            
            is_peak = 17 <= current_hour <= 22
            rate = 25.0 if is_peak else 15.0
            cost = dem * rate

            c_main = "#28a745"
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
                    else: st.success("System Normal")
                
                pdf_bytes = create_pdf(dem, risk, current_hour, current_temp, cost)
                st.download_button("📄 Save Log", data=pdf_bytes, file_name=f"live_log_{current_hour}h.pdf")
        else:
            st.error("Models not loaded.")

# --- PAGE 3: BATCH ANALYTICS ---
elif page == "Batch Analytics":
    st.title("📂 Batch Processing")
    up_file = st.file_uploader("Upload CSV", type=["csv", "txt"])
    
    if up_file and st.button("Process Batch File"):
        try:
            df = pd.read_csv(up_file)
            # Standardize columns
            df.columns = df.columns.str.strip().str.lower()
            
            required = ['hour', 'temperature', 'voltage', 'dayofweek']
            if all(col in df.columns for col in required):
                if 'regression' in models:
                    X = df[required]
                    df['Predicted_Demand'] = models['regression'].predict(X)
                    
                    st.success(f"Processed {len(df)} rows successfully!")
                    st.dataframe(df)
                    
                    if 'hour' in df.columns: 
                        st.markdown("### 📈 Batch Trend")
                        st.line_chart(df.set_index('hour')['Predicted_Demand'])
                else:
                    st.error("Regression model missing.")
            else:
                st.error(f"CSV must contain columns: {required}")
        except Exception as e:
            st.error(f"Error: {e}")