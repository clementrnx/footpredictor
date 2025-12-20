import streamlit as st
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="iTrOz Predictor | Low Entropy", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; color: #FFFFFF; }
    .stMetric { background-color: #111111; border: 1px solid #FFD700; padding: 10px; border-radius: 10px; }
    div[data-testid="stMetricValue"] > div { color: #FFD700 !important; }
    button { background-color: #FFD700 !important; color: black !important; font-weight: bold; width: 100%; border: none; padding: 10px; }
    .stTextInput>div>div>input, .stNumberInput>div>div>input { background-color: #111111; color: white; border: 1px solid #FFD700; }
    </style>
""", unsafe_allow_html=True)

def rho_correction(x, y, lambda_h, lambda_a, rho):
    if x == 0 and y == 0: return 1 - (lambda_h * lambda_a * rho)
    elif x == 0 and y == 1: return 1 + (lambda_h * rho)
    elif x == 1 and y == 0: return 1 + (lambda_a * rho)
    elif x == 1 and y == 1: return 1 - rho
    return 1

st.title("⚽ iTrOz Predictor")

st.sidebar.header("Parameters")
is_neutral = st.sidebar.checkbox("Neutral Ground")
rho = st.sidebar.slider("Entropy Adjustment (Rho)", -0.2, 0.2, -0.05)
home_adv = 1.0 if is_neutral else 1.10 

col1, col2 = st.columns(2)
with col1:
    h_name = st.text_input("Home Team", "FC Barcelona")
    h_att = st.number_input("Home Attack", value=2.1, step=0.1)
    h_def = st.number_input("Home Defense", value=0.9, step=0.1)

with col2:
    a_name = st.text_input("Away Team", "Villarreal")
    a_att = st.number_input("Away Attack", value=1.4, step=0.1)
    a_def = st.number_input("Away Defense", value=1.2, step=0.1)

if st.button("RUN ANALYTICS"):
    mu_h = h_att * a_def * home_adv
    mu_a = a_att * h_def * (1/home_adv)

    max_g = 9
    matrix = np.zeros((max_g, max_g))

    for x in range(max_g):
        for y in range(max_g):
            p_h = poisson.pmf(x, mu_h)
            p_a = poisson.pmf(y, mu_a)
            matrix[x, y] = p_h * p_a * rho_correction(x, y, mu_h, mu_a, rho)

    matrix /= matrix.sum()

    win_h = np.sum(np.tril(matrix, -1)) * 100
    draw = np.sum(np.diag(matrix)) * 100
    win_a = np.sum(np.triu(matrix, 1)) * 100

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric(h_name, f"{win_h:.1f}%")
    c2.metric("Draw", f"{draw:.1f}%")
    c3.metric(a_name, f"{win_a:.1f}%")

    best_score = np.unravel_index(matrix.argmax(), matrix.shape)
    st.success(f"🎯 Prediction: {best_score[0]} - {best_score[1]} ({matrix.max()*100:.1f}%)")
