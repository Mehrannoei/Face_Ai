import tensorflow as tf
import numpy as np
from PIL import Image

# بارگذاری مدل
model = tf.keras.models.load_model("model/skin_model.h5")

CLASSES = [
    "acne",
    "hyperpigmentation",
    "redness",
    "clear"
]

def analyze_skin(face_img):
    # تبدیل تصویر به فرمت مدل
    img = Image.fromarray(face_img)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]

    return dict(zip(CLASSES, preds.tolist()))
