from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="โปรเจคการจำแนกข้อมูลดอกไม้",
    page_icon="🌸",
    layout="wide"
)

# ----------------- Modern Floral CSS Styling -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

/* Main app background with delicate floral pastel gradients */
.stApp {
    background: linear-gradient(135deg, #FFF0F3 0%, #FDF4F5 30%, #F5F0FB 70%, #EFF8F5 100%);
    background-attachment: fixed;
    font-family: 'Prompt', sans-serif;
    color: #4A4A4A;
}

/* Main Header */
h1 {
    font-family: 'Prompt', sans-serif !important;
    color: #9D4067 !important;
    text-align: center;
    font-weight: 700 !important;
    padding-top: 10px;
    padding-bottom: 20px;
    letter-spacing: 0.5px;
    text-shadow: 1px 2px 4px rgba(157, 64, 103, 0.15);
}

/* Sub-headers inside flower cards */
[data-testid="column"] h2 {
    font-family: 'Prompt', sans-serif !important;
    color: #8C2F5E !important;
    text-align: center;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    margin-bottom: 12px;
}

/* Modern Glassmorphic Card for the 3 flower columns */
[data-testid="column"] {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 20px 16px;
    box-shadow: 0 10px 25px rgba(235, 131, 163, 0.15);
    border: 1px solid rgba(255, 214, 226, 0.7);
    transition: all 0.35s ease;
}

[data-testid="column"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 32px rgba(224, 87, 128, 0.25);
    border-color: #FFB3C1;
}

/* Modern styling for flower images */
[data-testid="stImage"] img {
    border-radius: 14px;
    object-fit: cover;
    width: 100%;
    height: 220px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
    transition: transform 0.3s ease;
}

[data-testid="stImage"] img:hover {
    transform: scale(1.02);
}

/* Dataframe container */
[data-testid="stDataFrame"] {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(8px);
    border-radius: 16px;
    padding: 10px;
    box-shadow: 0 8px 20px rgba(255, 182, 193, 0.15);
    border: 1px solid rgba(255, 214, 226, 0.6);
}

/* Modern Floral Buttons */
.stButton > button {
    background: linear-gradient(135deg, #FF758F 0%, #FF4D6D 100%) !important;
    color: white !important;
    font-family: 'Prompt', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border-radius: 30px !important;
    border: none !important;
    padding: 0.6rem 2.2rem !important;
    box-shadow: 0 8px 20px rgba(255, 77, 109, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 26px rgba(255, 77, 109, 0.45) !important;
    background: linear-gradient(135deg, #FF8FA3 0%, #E04360 100%) !important;
    color: white !important;
}

.stButton > button:active {
    transform: translateY(1px) !important;
}

/* Inputs & Sliders Cards */
.stSlider, .stNumberInput {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(6px);
    padding: 14px 20px;
    border-radius: 16px;
    border: 1px solid rgba(255, 214, 226, 0.6);
    margin-bottom: 14px;
    box-shadow: 0 6px 16px rgba(255, 182, 193, 0.1);
}

div[data-baseweb="slider"] {
    margin-top: 10px;
}

div[data-baseweb="input"] {
    border-radius: 12px !important;
    border-color: #FFB3C1 !important;
}

.stMarkdown p {
    font-family: 'Prompt', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ----------------- Title Header -----------------
st.header("🌸 โปรเจคการจำแนกข้อมูลดอกไม้ 🌸")
#st.image("./img/kairung022.jpg")

# ----------------- 3 Species Showcase -----------------
col1, col2, col3 = st.columns(3)

with col1:
   st.header("Versicolor")
   st.image("./img/iris1.jpg")

with col2:
   st.header("Verginiga")
   st.image("./img/iris2.jpg")

with col3:
   st.header("Setosa")
   st.image("./img/iris3.jpg")

# ----------------- Data Section -----------------
html_7 = """
<div style="background: linear-gradient(135deg, #FF758F 0%, #FF8FA3 50%, #FFB3C1 100%); padding: 16px 20px; border-radius: 18px; box-shadow: 0 8px 22px rgba(255, 117, 143, 0.3); margin: 24px 0 16px 0; text-align: center;">
    <h4 style="color: white; margin: 0; font-family: 'Prompt', sans-serif; font-weight: 600; letter-spacing: 0.5px;">🌸 สถิติข้อมูลดอกไม้ 🌸</h4>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
dt = pd.read_csv("./data/iris.csv")
st.write(dt.head(10))

dt1 = dt['petallength'].sum()
dt2 = dt['petalwidth'].sum()
dt3 = dt['sepallength'].sum()
dt4 = dt['sepalwidth'].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   #st.write(dt.head(10))
   st.bar_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")

# ----------------- Prediction Section -----------------
html_8 = """
<div style="background: linear-gradient(135deg, #A06CD5 0%, #B185DB 50%, #FF99C8 100%); padding: 16px 20px; border-radius: 18px; box-shadow: 0 8px 22px rgba(160, 108, 213, 0.3); margin: 28px 0 16px 0; text-align: center;">
    <h4 style="color: white; margin: 0; font-family: 'Prompt', sans-serif; font-weight: 600; letter-spacing: 0.5px;">🌺 ทำนายข้อมูล 🌺</h4>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")


pt_len = st.slider("กรุณาเลือกข้อมูล petal.length")
pt_wd = st.slider("กรุณาเลือกข้อมูล petal.width")

sp_len = st.number_input("กรุณาเลือกข้อมูล sepal.length")
sp_wd = st.number_input("กรุณาเลือกข้อมูล sepal.width")

if st.button("ทำนายผล"):
   #st.write("ทำนาย")
   #dt = pd.read_csv("./data/iris.csv") 

   X = dt.drop('variety', axis=1)
   y = dt.variety   
   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  

   x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]]) #ข้อมูลใหม่สำหรับทำนาย

   st.write(Knn_model.predict(x_input))   
   out=Knn_model.predict(x_input)

   if out[0] == 'Setosa':
    st.image("./img/iris1.jpg")
   elif out[0] == 'Versicolor':       
    st.image("./img/iris2.jpg")
   else:
    st.image("./img/iris3.jpg")
else:
    st.write("ไม่ทำนาย")
