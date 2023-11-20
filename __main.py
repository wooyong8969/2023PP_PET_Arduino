import serial
import time
import numpy as np
from dataProcessing import load_data, separate_data_by_type, data_to_numpy, calculate_csp
from SVMmodel import load_model, predict_input

# 모델 및 CSP 필터 로드
model_file = 'D:\\W00Y0NG\\PRGM2\\PP_PET_Project\\svm_model.joblib'
csp_filter_file = 'D:\\W00Y0NG\\PRGM2\\PP_PET_Project\\csp_filter.npy'
model = load_model(model_file)
B = np.load(csp_filter_file)

# 시리얼 포트 설정
py_serial = serial.Serial(
    port = "com4",  # 시리얼 포트
    baudrate = 9600,  # 통신 속도
    timeout = 1
)

# 아두이노에서 데이터 읽기
while True:
    if py_serial.readable():
        # 분광 센서 값은 아두이노 시리얼 모니터에서 각 값이 공백을 기준으로 구분되어 출력됐다 가정
        # ex) 40334 59925 59753 61173 62020 62094 61655 58229 62479 3761
        input_data = py_serial.readline()
        input_data = input_data.decode().strip()  # 문자열 변환 및 공백 제거
        x_data = np.fromstring(input_data, sep=',').reshape(1, -1)
        
        # CSP 적용
        x_csp = x_data.dot(B)
        x_reduced = x_csp[:, 0:2]

        # 예측 수행
        prediction = predict_input(model, x_reduced)
        prediction_str = "PP" if prediction[0] == 1 else "PET"

        print("예측 결과:", prediction_str)

        # 아두이노로 예측 결과 전송
        py_serial.write(prediction_str.encode())

    # 통신 오류 방지 대기
    time.sleep(0.1)
