import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
from pathlib import Path


st.set_page_config(
    page_title="SowEasy Crop Recommender",
    page_icon="🌾",
    layout="wide"
)


@st.cache_resource(show_spinner=False)
def train_models():
    df = pd.read_csv('Crop_recommendation.csv')
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    templates = {
        'Logistic Regression': LogisticRegression(random_state=2),
        'Decision Tree': DecisionTreeClassifier(criterion='entropy', random_state=2, max_depth=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=20, random_state=0),
    }

    models = {}
    performance = {}
    for name, estimator in templates.items():
        fitted = estimator.fit(X_train, y_train)
        models[name] = fitted
        performance[name] = accuracy_score(y_test, fitted.predict(X_test))

    return models, performance

def classify(answer):
    return answer[0]+" is the best crop for cultivation here."


def main():
    models, performance = train_models()

    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;600&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background: radial-gradient(circle at top, #14213d, #050910 55%);
        color: #f4f6fb;
    }
    .hero-card {
        background: rgba(7, 17, 31, 0.8);
        border-radius: 24px;
        padding: 32px;
        margin-bottom: 24px;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 20px 45px rgba(5, 10, 25, 0.6);
    }
    .hero-card h1 {
        margin: 0;
        color: #fdfdfd;
    }
    .hero-card p {
        color: #d2d9f8;
        font-size: 1rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        padding: 18px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }
    .metric-card h2 {
        color: #a3d9ff;
        margin: 0;
        font-size: 2rem;
    }
    .metric-card span {
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.75rem;
        color: #c5c9f5;
    }
    .prediction-card {
        background: rgba(7, 17, 31, 0.9);
        border-radius: 24px;
        padding: 30px;
        border: 1px solid rgba(255,255,255,0.07);
        box-shadow: 0 15px 40px rgba(0,0,0,0.35);
    }
    .stButton>button {
        background: linear-gradient(120deg, #00b894, #0984e3);
        color: #fff;
        border: none;
        border-radius: 16px;
        padding: 12px 20px;
        font-size: 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        box-shadow: 0 12px 20px rgba(0, 184, 148, 0.35);
    }
    .stSidebar {
        background: #050b16;
        padding: 25px 20px;
    }
    .sidebar-card h3 {
        color: #e5ecff;
        margin-top: 0;
    }
    .sidebar-card p {
        color: #98a4c9;
        font-size: 0.85rem;
    }
    .sidebar-card {
        background: rgba(255,255,255,0.03);
        border-radius: 18px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 18px;
    }
    label[data-baseweb="label"] {
        font-weight: 500 !important;
        color: #f4f6fb !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    image_path = Path('C:/Users/lenovo/Downloads/MLOPS TUBES/WhatsApp Image 2025-12-15 at 16.31.00_1c0a35a3.jpg')
    if image_path.exists():
        image = Image.open(image_path)
    else:
        image = None

    st.markdown(
        """
        <div class='hero-card'>
            <h1>Implementasi Workflow MLOps pada Sistem Rekomendasi Tanaman Berbasis Machine Learning</h1>
            <p>Prediksi tanaman paling cocok berdasar nutrisi tanah, cuaca mikro, dan pola hujan.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if image:
        st.image(image, use_container_width=True)
    else:
        st.info('Tambahkan hero image agar tampilan semakin kaya visual.')

    with st.sidebar:
        st.markdown("<div class='sidebar-card'><h3>Model Inference</h3><p>Pilih algoritma favorit dan masukkan parameter agronomi.</p></div>", unsafe_allow_html=True)
        activities=['Naive Bayes','Logistic Regression','Decision Tree','Random Forest']
        option=st.selectbox("Engine yang digunakan",activities)

        st.markdown("<div class='sidebar-card'><h3>Sensor Lapangan</h3><p>Rentang input menjaga data tetap valid.</p></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            sn=st.number_input('Nitrogen (N) • kg/ha', 0.0, 150.0, 60.0, step=1.0)
            pk=st.number_input('Kalium (K) • kg/ha', 0.0, 210.0, 50.0, step=1.0)
            phu=st.number_input('Kelembapan • %', 0.0, 100.0, 60.0, step=1.0)
        with col2:
            sp=st.number_input('Fosfor (P) • kg/ha', 0.0, 150.0, 60.0, step=1.0)
            pt=st.number_input('Suhu • °C', 0.0, 50.0, 25.0, step=0.5)
            pPh=st.number_input('pH Tanah', 0.0, 14.0, 6.5, step=0.1)
        pr=st.number_input('Curah Hujan • mm', 0.0, 300.0, 100.0, step=1.0)

    st.markdown("### Insight Akurasi Model")
    metric_cols = st.columns(len(performance))
    for (name, score), col in zip(performance.items(), metric_cols):
        col.markdown(f"<div class='metric-card'><span>{name}</span><h2>{score*100:.2f}%</h2></div>", unsafe_allow_html=True)

    st.markdown("### Konsol Rekomendasi Tanaman")
    inputs=[[sn,sp,pk,pt,phu,pPh,pr]]
    with st.container():
        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        if st.button('Diagnosa Lahan', use_container_width=True):
            selected_model = models[option]
            suggestion = classify(selected_model.predict(inputs))
            st.success(suggestion)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__=='__main__':
    main()
