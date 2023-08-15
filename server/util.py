import cv2
import json
import numpy as np
import base64
import joblib
from wavelet import w2d

__model = None
__class_name_to_number = {}
__class_number_to_name = {}

def get_img_base64_string(base64str):
    encoded_data = base64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_img(based64str, img_path):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

    if img_path:
        img = cv2.imread(img_path)
    else:
        img = get_img_base64_string(based64str)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes)==2:
            cropped_faces.append(roi_frame)

    return cropped_faces

def load_saved_artifacts():
    print('loading saved artifacts...')
    global __model, __class_name_to_number, __class_number_to_name

    with open('./artifacts/class_dictionary.json', 'r') as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)

def num_to_name(num):
    return __class_number_to_name[num]

def classify_img(base64str, img_path=None):
    image = get_cropped_img(base64str, img_path)
    result = []

    for img in image:
        scaled_img = cv2.resize(img, (32,32))
        img_har = w2d(img, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32,32))
        combined_img = np.vstack((scaled_img.reshape(32*32*3, 1), scaled_img_har.reshape(32*32, 1))) # reshape về 2-D

        len_img = 32*32*3 + 32*32
        final = combined_img.reshape(1, len_img).astype(float)

        result.append({
            'class': num_to_name(__model.predict(final)[0]), # vì predict dùng 2-D mà func chỉ lấy 1 item nên lấy index 0
            'class_probability': np.round(__model.predict_proba(final) * 100, 2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })
    return result


if __name__ == '__main__':
    load_saved_artifacts()
    result = classify_img(None, './test_images/virat1.jpg')
    print(result)
