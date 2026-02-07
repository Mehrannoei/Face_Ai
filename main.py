from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

app = FastAPI()

# اجازه ارتباط فرانت‌اند
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# مسیر فرانت
frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")

# سرو کردن فایل‌ها و استاتیک‌ها
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# بارگذاری مدل AI
model = tf.keras.models.load_model("model/skin_model.h5")

# کلاس‌های پوست
classes = ["acne", "hyperpigmentation", "redness", "clear"]

# توضیح و پیشنهاد برای هر کلاس
recommendations = {
    "acne": {
        "desc": "پوست مستعد جوش است و نیاز به کنترل چربی و ضد التهاب دارد.",
        "treatment": "ژل بنزوئیل پراکساید یا سالیسیلیک اسید، شستشو با شوینده ملایم"
    },
    "hyperpigmentation": {
        "desc": "لک‌های پوستی ناشی از آفتاب یا جای جوش دیده می‌شود.",
        "treatment": "کرم ویتامین C، نیاسینامید، ضدآفتاب"
    },
    "redness": {
        "desc": "قرمزی پوست ممکن است به دلیل التهاب یا حساسیت باشد.",
        "treatment": "کرم زینک اکساید، آلوئه‌ورا، پرهیز از مواد تحریک‌کننده"
    },
    "clear": {
        "desc": "پوست سالم و بدون مشکل خاص تشخیص داده شد.",
        "treatment": "مرطوب‌کننده ملایم و ضدآفتاب روزانه"
    }
}

# پیش‌پردازش تصویر برای مدل
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Endpoint اصلی تحلیل پوست
@app.post("/analyze")
async def analyze_skin(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processed = preprocess_image(image)

    prediction = model.predict(processed)
    confidence = float(np.max(prediction))
    class_index = int(np.argmax(prediction))
    class_name = classes[class_index]

    if confidence < 0.25:
        class_name = "clear"

    response = {
        "diagnosis": class_name,
        "confidence": confidence,
        "description": recommendations[class_name]["desc"],
        "recommendation": recommendations[class_name]["treatment"]
    }

    if 0.4 <= confidence <= 0.6 and class_name != "clear":
        response["note"] = "ممکن است پوست شما نشانه‌های خفیف دیگری نیز داشته باشد. بررسی تخصصی توصیه می‌شود."

    return response

# روت اصلی: سرو کردن index.html فرانت
@app.get("/")
async def root():
    return FileResponse(os.path.join(frontend_path, "index.html"))