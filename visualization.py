import matplotlib.pyplot as plt
import numpy as np

# CSP 적용 결과 시각화
def plot_data(pp_data, pet_data):
    plt.figure()
    plt.plot(pp_data[:, 0], pp_data[:, 1], 'b*', label='PP')
    plt.plot(pet_data[:, 0], pet_data[:, 1], 'ro', label='PET')
    plt.title('CSP of PET and PP Data')
    plt.legend()
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

# SVM 학습 결과 시각화
def plot_decision_boundary(clf, X, Y):
    x_min, x_max = X[:, 0].min() - 50, X[:, 0].max() + 50
    y_min, y_max = X[:, 1].min() - 50, X[:, 1].max() + 50
    xx, yy = np.meshgrid(np.arange(x_min, x_max),
                         np.arange(y_min, y_max))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('SVM Result')
    plt.show()
