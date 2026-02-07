import cv2
import numpy as np

# بارگذاری مدل تشخیص صورت
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face(image_bytes):
    # تبدیل bytes به تصویر
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return None

    # تغییر سایز برای دقت بهتر
    img = cv2.resize(img, (0, 0), fx=0.75, fy=0.75)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,      # حساس‌تر
        minNeighbors=3,       # کمتر = راحت‌تر تشخیص می‌ده
        minSize=(60, 60)      # صورت خیلی کوچیک رد می‌شه
    )

    if len(faces) == 0:
        return None

    # بزرگ‌ترین صورت
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_img = img[y:y+h, x:x+w]

    return face_img
