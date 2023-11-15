import pandas as pd
import numpy as np

# csv data 불러오기
def load_data(filename):
    return pd.read_csv(filename)

# PP와 PET를 data를 분류 (라벨 열은 제외하고, 오직 값만을 dataFrame으로 저장)
def separate_data_by_type(data, type_col='Type'):
    pp_data = data[data[type_col] == 'PP'].drop([type_col], axis=1)
    pet_data = data[data[type_col] == 'PET'].drop([type_col], axis=1)
    return pp_data, pet_data

# dataFrame을 array로 변환
def data_to_numpy(pp_data, pet_data):
    pp_arr = pp_data.to_numpy()
    pet_arr = pet_data.to_numpy()
    return pp_arr, pet_arr

# CSP 적용
def calculate_csp(pp_arr, pet_arr):
    C_pp = np.cov(pp_arr, rowvar=False)
    C_pet = np.cov(pet_arr, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(C_pp).dot(C_pet))
    sorted_indices = np.argsort(eigvals)[::-1]
    B = eigvecs[:, sorted_indices]
    pp_result = pp_arr.dot(B)
    pet_result = pet_arr.dot(B)
    return pp_result, pet_result, B
