from utils import *
from sklearn.model_selection import train_test_split
print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

path = 'data_collection'
data = importDataInfo(path)

data = balanceData(data, display=False)

imagesPath, steerings, throttles = loadData(path, data)
print(imagesPath[0], steerings[0], throttles[0])

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, list(zip(steerings, throttles)), test_size=0.2, random_state=10)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

model = createModel()
model.summary()

history = model.fit(batchGen(xTrain, yTrain, 100, 1), steps_per_epoch=250, epochs=10,
                    validation_data=batchGen(xVal, yVal, 100, 0), validation_steps=200)

model.save('model.h5')
print('model saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim([0, 1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
