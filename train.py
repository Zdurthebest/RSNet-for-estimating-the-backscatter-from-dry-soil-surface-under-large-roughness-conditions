

'''
   train RSNet
   2023.2.27
'''

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from model import doublelstm
from utils import readdata
import pandas as pd
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# r_shape = (2, 1)
# s_shape = (4, 1)
# y_shape = 2

###########################################
#    read radar/surface parameter         #
#    read backscattering coefficients     #
###########################################
r_path = 'radar_data.mat'
s_path = 'sur_data.mat'
y_path = 'sig_data.mat'

r_train, s_train, y_train, r_val, s_val, y_val, _, _, _ = readdata(r_path, s_path, y_path)

y_train = y_train / -50
y_val = y_val / -50

# print(y_train.shape)

##########################################
#             load model                 #
##########################################
model = doublelstm()

model.compile(loss='mse', optimizer=Adam(lr=0.001, beta_1=0.9), metrics=['accuracy'])

print('**********************************************************')
print('******************  Start training  *********************')
print('**********************************************************')
batch = 64
epochs = 1000
checkPoint = ModelCheckpoint("bestModel.hdf5", monitor='val_loss', save_best_only=True, period=1)
history = model.fit([r_train, s_train], y_train, batch_size=batch, epochs=epochs, shuffle=True,
                    verbose=1, validation_data=([r_val, s_val], y_val), callbacks=checkPoint)

print('**********************************************************')
print('******************  Finish training  *********************')
print('**********************************************************')


pd.DataFrame(history.history).to_csv('loss.csv', index=False)

#########################
####   Get history   ####
####   Plot loss     ####
#########################
fig = plt.figure()
loss = history.history['loss']
ep = range(len(loss))
plt.plot(ep, loss, 'g', label='Train loss')
plt.show()

