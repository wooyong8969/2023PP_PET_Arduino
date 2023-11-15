from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load

# train / test 데이터 분류
def train_test_split_data(x, y, test_size=0.2, random_state=42):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

# 선형 svm 모델 학습
def train_svm(x_train, y_train, kernel='linear', C=1):
    model = SVC(kernel=kernel, C=C)
    model.fit(x_train, y_train)
    return model

# svm 성능 평가
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# 단일 데이터의 예측 결과 반환
def predict_input(model, x_data):
    pred = model.predict(x_data)
    y_pred = "PP" if pred[0] == 1 else "PET"
    return y_pred

# 학습한 모델을 파일로 저장
def save_model(model, filename):
    dump(model, filename)

# 파일로부터 모델 로드
def load_model(filename):
    return load(filename)