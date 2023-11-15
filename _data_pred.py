from dataProcessing import load_data, separate_data_by_type, data_to_numpy, calculate_csp
from SVMmodel import load_model, predict_input
import numpy as np

model_file = 'D:\W00Y0NG\PRGM2\PP_PET_Project\svm_model.joblib'
csp_filter_file = 'D:\W00Y0NG\PRGM2\PP_PET_Project\csp_filter.npy'

loaded_model = load_model(model_file)
B = np.load(csp_filter_file)

x_data = np.array([40334, 59925, 59753, 61173, 62020, 62094, 61655, 58229, 62479, 37610]).reshape(1, -1)
x_csp = x_data.dot(B)
x_reduced = x_csp[:, 0:2]

prediction = predict_input(loaded_model, x_reduced)

print("예측 결과:", prediction)