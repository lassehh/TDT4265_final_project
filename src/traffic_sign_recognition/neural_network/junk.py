from read_traffic_signs import *
import matplotlib.pyplot as plt

[X_test, Y_test] = readTestingTrafficSigns('GTSRB/Final_Testing/Images')
print(len(Y_test), len(X_test))
plt.imshow(X_test[42])
plt.show()