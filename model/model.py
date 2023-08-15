import numpy as np
import cv2
import pywt
from matplotlib import pyplot as plt
import os


face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')


''' DATA CLEANING GET CROPPED IMG '''
def get_cropped_img_2_eyes(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 2:
            return roi_frame

path_data = './dataset/players/'
img_dirs = []
for entry in os.scandir(path_data):
    if entry.is_dir():
        img_dirs.append(entry.path)
print(img_dirs)

path_cropped = './dataset/cropped/'
cropped_dirs = []
celeb_dict = {}

for dir in img_dirs:
    count = 1
    celeb_name = dir.split('/')[-1]
    celeb_dict[celeb_name] = []

    for entry in os.scandir(dir):  # scan img each dir
        roi_frame = get_cropped_img_2_eyes(entry.path)
        if roi_frame is not None:
            cropped_folder = path_cropped + celeb_name
            cropped_dirs.append(cropped_folder)

            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)

            cropped_file_name = celeb_name + str(count) + '.png'
            cropped_file_path = cropped_folder + '/' + cropped_file_name

            cv2.imwrite(cropped_file_path, roi_frame)
            celeb_dict[celeb_name].append(cropped_file_path)
            count+=1

''' FEATURE ENGINEERING (EXTRACT IMPORTANT FEATURE) '''
def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversion
    #convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    #convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    #compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    #Process coefficients
    coeffs_h = list(coeffs)
    coeffs_h[0] *= 0

    #Reconstruction
    imArray_h = pywt.waverec2(coeffs_h, mode)
    imArray_h *= 255
    imArray_h = np.uint8(imArray_h)

    return imArray_h

roi_frame = get_cropped_img_2_eyes('./test_images/sharapova1.jpg')
img_har = w2d(roi_frame, 'db1', 5)
print(roi_frame.shape, img_har.shape)
plt.imshow(img_har, cmap='gray') # extracting important facial features where are white as eyes, nose, lip


# lọc những img 
celeb_dict = {}
for img_dir in cropped_dirs:
    celebrity_name = img_dir.split('/')[-1]
    file_list = []
    for entry in os.scandir(img_dir):
        file_list.append(entry.path)
    celeb_dict[celebrity_name] = file_list

# đánh dấu number cho các celeb name
class_dict = {}
count = 0
for celeb_name in celeb_dict.keys():
    class_dict[celeb_name] = count
    count += 1
print(class_dict)
# y là string nên y bắt buộc phải chuyển qua number
x, y = [], []
for celeb_name, train_file in celeb_dict.items():
    for train_img in train_file:
        img = cv2.imread(train_img)
        scaled_img = cv2.resize(img, (32,32))
        img_har = w2d(img, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32,32))
        combined_img = np.vstack((scaled_img.reshape(32*32*3,1), scaled_img_har.reshape(32*32,1))) # reshape về 2-D
        x.append(combined_img)
        y.append(class_dict[celeb_name])

print(len(x))
print(len(x[0])) # = 32*32*3 + 32*32 = 4096
x = np.array(x).reshape(len(x), 4096).astype(float)

''' TRAIN MODEL '''
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib
import json

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=10))])
pipe.fit(x_train, y_train)
print(pipe.score(x_test, y_test))
print(classification_report(y_test, pipe.predict(x_test)))

model_params = {
    'svm': {
        'model': SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1,10,20],
            'svc__kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [1,5,10] #ta dùng pipeline phải khai báo param đúng cú pháp như vậy nếu ko error tương tự line 172-173, 185
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'logisticregression__C': [1,5,10]
        }
    }
}

scores = []
best_estimators = {}
for key, item in model_params.items():
    pipeline = make_pipeline(StandardScaler(), item['model'])
    clf = GridSearchCV(pipeline, item['params'], cv=5, return_train_score=False)
    clf.fit(x_train, y_train)
    scores.append({
        'model': key,
        'best_score': clf.best_score_,
        'best_param': clf.best_params_
    })
    best_estimators[key] = clf.best_estimator_
df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_param'])
print(df)

print(best_estimators['svm'].score(x_test, y_test))
print(best_estimators['random_forest'].score(x_test, y_test))
print(best_estimators['logistic_regression'].score(x_test, y_test))

best_clf = best_estimators['svm']
joblib.dump(best_clf, 'saved_model.pkl') # save as pickle file

with open('class_dictionary.json', 'w') as f:
    f.write(json.dumps(class_dict))