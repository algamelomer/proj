print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

import base64
from io import BytesIO
from PIL import Image
import cv2
 
#### FOR REAL TIME COMMUNICATION BETWEEN CLIENT AND SERVER
sio = socketio.Server()
#### FLASK IS A MICRO WEB FRAMEWORK WRITTEN IN PYTHON
app = Flask(__name__)  # '__main__'
 
maxSpeed = 10
 
def preProcess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
 
 
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    
    # Pre-process image
    processed_image = preProcess(image)
    print("Processed Image Shape:", processed_image.shape)  # Debug: Check processed image shape
    
    # Reshape for model prediction
    image_input = np.array([processed_image])
    
    # Predict steering angle
    steering = float(model.predict(image_input))
    
    # Calculate throttle
    throttle = 1.0 - speed / maxSpeed
    
    print(f'Predicted Steering: {steering}, Throttle: {throttle}, Speed: {speed}')  # Debug: Print predictions
    
    # Send control commands to simulator
    sendControl(steering, throttle)

# Print model summary for detailed information
    print(model.summary())

 
 
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)
 
 
def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })
 
 
if __name__ == '__main__':
    custom_objects = {'mse': MeanSquaredError()}
    model = load_model('1st.h5', custom_objects=custom_objects)
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Print model information for verification
    print("Model Input Shape:", model.input_shape)
    print("Model Output Shape:", model.output_shape)
    
    app = socketio.Middleware(sio, app)
    ### LISTEN TO PORT 4567
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
