

'''
  工具类
  zhudong
  2023.2.27
'''


import numpy as np
import h5py


r_shape = (2, 1)
s_shape = (4, 1)
y_shape = 2

train_rate = 0.7
val_rate = 0.2

def readdata(r_path, s_path, y_path):

    temp_r = h5py.File(r_path)
    temp_s = h5py.File(s_path)
    temp_y = h5py.File(y_path)
    data_r = np.array(temp_r['r_para'])
    data_s = np.array(temp_s['s_para'])
    data_y = np.array(temp_y['y'])

    data_r = data_r.reshape((data_r.shape[0], r_shape[0], r_shape[1]))
    data_s = data_s.reshape((data_s.shape[0], s_shape[0], s_shape[1]))
    data_y = data_y.reshape((data_y.shape[0], y_shape))

    ##########################################
    #    get train/test/validate dataset     #
    #    train 0.7 val 0.2 test 0.1          #
    ##########################################
    num_train = int(data_r.shape[0] * train_rate)
    num_val = int(data_r.shape[0] * val_rate)

    r_train = data_r[0:num_train, ::]
    s_train = data_s[0:num_train, ::]
    y_train = data_y[0:num_train, :]
    r_val = data_r[num_train:(num_train + num_val), ::]
    s_val = data_s[num_train:(num_train + num_val), ::]
    y_val = data_y[num_train:(num_train + num_val), :]
    r_test = data_r[(num_train + num_val):-1, ::]
    s_test = data_s[(num_train + num_val):-1, ::]
    y_test = data_y[(num_train + num_val):-1, :]

    return r_train, s_train, y_train, r_val, s_val, y_val, r_test, s_test, y_test

