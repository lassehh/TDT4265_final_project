import load_traffic_signs
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

[X_test, Y_test] = load_traffic_signs.training('GTSRB/Final_Training/Images')
print(len(Y_test), len(X_test))
#plt.imshow(X_test[42])
#plt.show()

#img = Image.open("someimg.png")
print(len(X_test[42][0]), len(X_test[42][1]))
img = Image.fromarray(X_test[42])
img = img.resize((20,20), Image.HAMMING)
print(img.size)
plt.imshow(img)
plt.show()