import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="AI Skin Analysis",
    page_icon="🧠",
    layout="centered"
)

# ------------------------------
# CSS (فونت + بک‌گراند + کارت‌ها)
# ------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Vazirmatn', sans-serif;
}

.main {
    background-image: url("https://www.transparenttextures.com/patterns/cubes.png");
    background-color: #f4f6fb;
}

/* Welcome box */
.welcome-box {
    background: linear-gradient(135deg, #ffe259, #ffa751);
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    color: #333;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    margin-bottom: 30px;
}

/* Center image */
.center-img {
    display: flex;
    justify-content: center;
    margin-top: 15px;
}

/* Cards */
.card {
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    margin-top: 15px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    font-size: 16px;
}

.card-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 8px;
}

.blue { background: #e3f2fd; }
.green { background: #e8f5e9; }
.purple { background: #f3e5f5; }
.red { background: #fdecea; }

.footer {
    text-align: center;
    margin-top: 50px;
    color: gray;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Welcome
# ------------------------------
st.markdown("""
<div class="welcome-box">
  <h1>👋 خوش آمدید</h1>
  <p>سیستم هوش مصنوعی تحلیل پوست صورت</p>
  <p>تصویر صورت خود را آپلود کنید تا بررسی شود</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Load model
# ------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/skin_model.h5")

model = load_model()

# ------------------------------
# Info data
# ------------------------------
CLASS_NAMES = ["acne", "clear", "hyperpigmentation", "redness"]

SKIN_INFO = {
    "acne": {
        "title": "آکنه (جوش پوستی)",
        "desc": "آکنه به دلیل بسته شدن منافذ پوست با چربی و سلول‌های مرده ایجاد می‌شود.",
        "rec": "شست‌وشوی منظم، عدم دستکاری جوش‌ها و مراجعه به پزشک در صورت شدت توصیه می‌شود.",
        "color": "red"
    },
    "redness": {
        "title": "قرمزی پوست",
        "desc": "قرمزی پوست می‌تواند ناشی از التهاب، حساسیت یا روزاسه باشد.",
        "rec": "پرهیز از عوامل تحریک‌کننده و استفاده از محصولات ملایم توصیه می‌شود.",
        "color": "purple"
    },
    "hyperpigmentation": {
        "title": "لک و تیرگی پوست",
        "desc": "لک‌های پوستی به دلیل افزایش تولید ملانین ایجاد می‌شوند.",
        "rec": "استفاده از ضدآفتاب و محصولات روشن‌کننده مفید است.",
        "color": "blue"
    },
    "clear": {
        "title": "پوست سالم",
        "desc": "پوست شما در وضعیت طبیعی و سالم قرار دارد.",
        "rec": "روتین مراقبتی فعلی خود را ادامه دهید.",
        "color": "green"
    }
}

# ------------------------------
# Upload
# ------------------------------
uploaded_file = st.file_uploader(
    "📤 آپلود تصویر صورت",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="center-img">', unsafe_allow_html=True)
    st.image(image, width=240)
    st.markdown('</div>', unsafe_allow_html=True)

    img = image.resize((224, 224))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    with st.spinner("🧠 در حال تحلیل تصویر..."):
        pred = model.predict(x, verbose=0)[0]

    idx = np.argmax(pred)
    label = CLASS_NAMES[idx]
    confidence = float(pred[idx])

    info = SKIN_INFO[label]

    # Diagnosis card
    st.markdown(f"""
    <div class="card {info['color']}">
        <div class="card-title">تشخیص</div>
        <p>{info['title']}</p>
    </div>
    """, unsafe_allow_html=True)
    # Confidence card
    st.markdown(f"""
    <div class="card blue">
        <div class="card-title">میزان اطمینان</div>
        <p>{confidence*100:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # Description card
    st.markdown(f"""
    <div class="card purple">
        <div class="card-title">توضیحات</div>
        <p>{info['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Recommendation card
    st.markdown(f"""
    <div class="card green">
        <div class="card-title">پیشنهاد مراقبتی</div>
        <p>{info['rec']}</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
<div class="footer">
Made with ❤️ by Mehran Noei
</div>
""", unsafe_allow_html=True)