import numpy as np
import matplotlib.pyplot as plt

number_parameters = 40
x1_para = np.linspace(-1,1, number_parameters)
x2_para = np.linspace(-1,1, number_parameters)

x1 , x2 =np.meshgrid(x1_para, x2_para)
def y_para(x1 , x2):
    y= np.sin((3.5*x1) + 1) * np.cos(5.5*x2) 
    return y
y = y_para(x1 , x2)

ax = plt.axes(projection='3d')
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")

ax.plot_surface(x1, x2, y)
ax
