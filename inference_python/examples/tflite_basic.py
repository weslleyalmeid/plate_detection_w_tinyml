import re
import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import ipdb

path_default = os.path.dirname(os.path.abspath(__file__))
LABEL_MAP_NAME = 'label_map.txt'
CUSTOM_MODEL_NAME = 'detect_model_int_convert.tflite'

paths = {
    'IMAGE_PATH_TEST': os.path.join(os.path.join(path_default, 'data'), 'test'),
    'OUTPUT_PATH': os.path.join(path_default, 'output'),
    'TFLITE_PATH':os.path.join(path_default, 'models', CUSTOM_MODEL_NAME),
 }


def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels

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


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results


def save_image(img, result, name_img, x_shape, y_shape):

  CAMERA_WIDTH = 320
  CAMERA_HEIGHT = 320

  ymin, xmin, ymax, xmax = result[0]['bounding_box']
  xmin = int(max(1,xmin * CAMERA_WIDTH))
  xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
  ymin = int(max(1, ymin * CAMERA_HEIGHT))
  ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))

  rect = cv2.rectangle(img,(xmin, ymin),(xmax, ymax),(0,255,0),3)

  cv2.imwrite(os.path.join(paths['OUTPUT_PATH'], name_img), cv2.resize(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB), (x_shape, y_shape)))


if __name__ == '__main__':

  # limiar de detecção e limiar de extração da licença
  detection_threshold = 0.8
  region_threshold = 0.55
  name_img = 'Cars403.png'
  IMAGE_PATH = os.path.join(paths['IMAGE_PATH_TEST'], name_img)

  # Realiza a identificação de placa em uma imagem
  img = cv2.imread(IMAGE_PATH)
  x_shape, y_shape, _ = img.shape

  img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (320,320))
  img = np.array(img)

  plt.imshow(img)

  interpreter = tf.lite.Interpreter(paths['TFLITE_PATH'])
  interpreter.allocate_tensors()
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  result = detect_objects(interpreter, img, detection_threshold)

  print(result)

  if result:
    save_image(img, result, name_img, x_shape, y_shape)
  else:
    print('não foi possível detectar!')

