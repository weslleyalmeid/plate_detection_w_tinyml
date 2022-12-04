import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import string
import ipdb
import glob
import shutil
import pandas as pd
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
CUSTOM_MODEL = os.path.join(MODELS_DIR, 'experiments', 'dynamic_range_quantization.tflite')
# IMAGE_DIR =  os.path.join(ROOT_DIR, 'custom_dataset')
IMAGE_DIR =  os.path.join(ROOT_DIR, 'test')


DEFAULT_ALPHABET = string.digits + string.ascii_uppercase


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold, modelo):
    """Returns a list of detection results, each a dictionary of object info."""

    if 'integer_only' in modelo:
        input_details = interpreter.get_input_details()[0]
        input_scale, input_zero_point = input_details["quantization"]
        image = (image / input_scale) + input_zero_point
        image = np.expand_dims(image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], image)

    else:
        set_input_tensor(interpreter, image)
    
    interpreter.invoke()
    
    boxes = get_output_tensor(interpreter, 1)
    classes = get_output_tensor(interpreter, 3)
    scores = get_output_tensor(interpreter, 0)
    count = int(get_output_tensor(interpreter, 2))


    # Get all output details
    if 'integer_only' in modelo:
        output_scale, output_zero_point = interpreter.get_output_details()[0]['quantization']
        scores = (scores[scores.argmax()] - output_zero_point) * output_scale

    else:
        boxes = get_output_tensor(interpreter, 1)
        classes = get_output_tensor(interpreter, 3)
        scores = get_output_tensor(interpreter, 0)
        count = int(get_output_tensor(interpreter, 2))

    result = {}

    if scores[0] >= threshold:
        result = {
            'bounding_box': boxes[0],
            'class_id': classes[0],
            'score': scores[0]
        }

    return result


def adjust_image(result, model_x_shape, model_y_shape):
    
    CAMERA_WIDTH = model_x_shape
    CAMERA_HEIGHT = model_y_shape
    ymin, xmin, ymax, xmax = result[0]['bounding_box']

    xmin = int(max(1, xmin * CAMERA_WIDTH))
    xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
    ymin = int(max(1, ymin * CAMERA_HEIGHT))
    ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))

    return xmin, ymin, xmax, ymax


if __name__ == '__main__':

    # df = pd.read_csv(os.path.join(ROOT_DIR, 'info.csv'))
    df_final = pd.DataFrame(columns = ['file', 'modelo', 'time', 'detect', 'score'])
    
    # limiar de detecção e limiar de extração da licença
    detection_threshold = 0.6
    
    # models = glob.glob(os.path.join(MODELS_DIR, 'experiments', '*.tflite'))
    models = glob.glob(os.path.join(MODELS_DIR, 'experiments', 'full_int8.tflite'))
    files = glob.glob(IMAGE_DIR + '/*.jpg')
    files.extend(glob.glob(IMAGE_DIR + '/*.png'))


    
    for model in models:
        modelo = model.split('/')[-1].split('.')[0]

        interpreter = tf.lite.Interpreter(model)
        interpreter.allocate_tensors()

        _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
        
        imagens = []
        results = []
        
        for imgs in files:
            
            file = imgs.split('/')[-1]
            print(file)

            img = cv2.imread(imgs)
            x_shape, y_shape, _ = img.shape

            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (input_width, input_height))
            img = np.array(img)
            
            start = time.process_time()
            result = detect_objects(interpreter, img, detection_threshold, modelo)
            end = time.process_time() - start
            
            if result:

                row = {
                    'file': file,
                    'modelo': modelo,
                    'time': end,
                    'detect': 1,
                    'score': result.get('score')
                }
                
            else:
                row = {
                    'file': file,
                    'modelo': modelo,
                    'time': end,
                    'detect': 0,
                    'score': 0
                }
            
    
            df_final = pd.concat([df_final, pd.DataFrame([row])], ignore_index=True)

    import ipdb; ipdb.set_trace()
    # df_final.to_csv('./experiments.csv', index=False)