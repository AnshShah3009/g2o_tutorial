import numpy as np

v20 = np.array([3.221425985179073, -6.845279749494726, 0.1247604949879307])
v21 = np.array([3.663371320004392, -6.739181318919993, 0.3617104429951685])

e20_21 = np.array([0.451713, 0.0502794, 0.23695])
theta_20 = v20[2]
del_x = e20_21[0]; del_y = e20_21[1]
Del_x =( del_x * np.cos(theta_20)) - (del_y * np.sin(theta_20))
Del_y =( del_y * np.cos(theta_20)) + (del_x * np.sin(theta_20))

print(v21[:2])
#print(v20[:2])
print(v20[:2] +  np.array([Del_x, Del_y]))
