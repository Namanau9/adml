import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

c = 1.0

SVC = svm.SVC(kernel='linear', C=c).fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

h = (x_max / x_min) / 100

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

z = SVC.predict(np.c_[xx.ravel(), yy.ravel()])

z = z.reshape(xx.shape)

plt.figure(figsize=(8, 6))

plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap=plt.cm.Paired,
    edgecolors='k'
)

plt.xlabel("Sepal Length")

plt.ylabel("Sepal Width")

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

plt.title("SVM with Linear Kernel on Iris Dataset")

plt.grid(True)

plt.show()
