from dataProcessing import load_data, separate_data_by_type, data_to_numpy, calculate_csp
from visualization import plot_data, plot_decision_boundary
from SVMmodel import train_test_split_data, train_svm, evaluate_model, save_model
import numpy as np

# 경로 설정
data_file = 'D:\W00Y0NG\PRGM2\PP_PET_Project\Data\output.csv'
model_file = 'D:\W00Y0NG\PRGM2\PP_PET_Project\svm_model.joblib'
csp_filter_file = 'D:\W00Y0NG\PRGM2\PP_PET_Project\csp_filter.npy'

data = load_data(data_file)

# CSP 적용
pp_data, pet_data = separate_data_by_type(data)
pp_arr, pet_arr = data_to_numpy(pp_data, pet_data)
pp_result, pet_result, B = calculate_csp(pp_arr, pet_arr)
np.save(csp_filter_file, B) # CSP 필터 저장

plot_data(pp_result, pet_result)

# SVM 구현 및 평가 _ 1이 PP, 0이 PET
x = np.concatenate([pp_result[:, 0:2], pet_result[:, 0:2]])
y = np.concatenate([np.ones(pp_arr.shape[0]), np.zeros(pet_arr.shape[0])])

x_train, x_test, y_train, y_test = train_test_split_data(x, y)
model = train_svm(x_train, y_train)
evaluate_model(model, x_test, y_test)
save_model(model, model_file) # SVM 모델 저장

plot_decision_boundary(model, x_train, y_train)
