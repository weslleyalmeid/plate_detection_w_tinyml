import re
import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import string
from pprint import pprint
import ipdb

path_default = os.path.dirname(os.path.abspath(__file__))
LABEL_MAP_NAME = 'label_map.txt'
CUSTOM_MODEL_NAME = 'detect_original.tflite'
# CUSTOM_MODEL_NAME = 'detect_model_int_convert.tflite'
# CUSTOM_MODEL_NAME = 'detect_model_uint8_convert.tflite'
# CUSTOM_MODEL_NAME = 'detect_model_normal_convert.tflite'
# CUSTOM_MODEL_NAME = 'detect_model_normal_convert_cli.tflite'
# CUSTOM_MODEL_NAME = 'detect_model_float16.tflite'
characterRecognition = tf.keras.models.load_model('./models/character_recognition.h5')

paths = { 
    'IMAGE_PATH_TEST': os.path.join(os.path.join(path_default, 'data'), 'test_br'),
    'OUTPUT_PATH': os.path.join(path_default, 'output'), 
    'TFLITE_PATH':os.path.join(path_default, 'models', CUSTOM_MODEL_NAME),
 }

############### Modelo De Detecção tflite

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


def detect_objects(interpreter, image, threshold, original):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details

  # TODO: modelo original vem nessa ordem
  if original:
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

  # TODO: meu modelo tflite Converter, nessa ordem
  else:
    scores = get_output_tensor(interpreter, 0)
    boxes = get_output_tensor(interpreter, 1)
    count = int(get_output_tensor(interpreter, 2))
    classes = get_output_tensor(interpreter, 3)

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


def detect_one_object(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details

  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)

  # TODO: quando rodar em realtime, lembrar de alterar esse cara para o comentado
  count = 1
  
  results = []
  for i in range(count):
    import ipdb; ipdb.set_trace()
    if scores >= threshold:
      result = {
          'bounding_box': boxes,
          'class_id': classes,
          'score': scores
      }
      results.append(result)
  return results
  


def save_image(img, result, name_img, x_shape, y_shape):

  CAMERA_WIDTH = 320
  CAMERA_HEIGHT = 320
  import ipdb; ipdb.set_trace()
  ymin, xmin, ymax, xmax = result[0]['bounding_box']
  xmin = int(max(1,xmin * CAMERA_WIDTH))
  xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
  ymin = int(max(1, ymin * CAMERA_HEIGHT))
  ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))

  rect = cv2.rectangle(img,(xmin, ymin),(xmax, ymax),(0,255,0),3)

  cv2.imwrite(os.path.join(paths['OUTPUT_PATH'], name_img), cv2.resize(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB), (x_shape, y_shape)))


############### Meu pre-process OCR

def ocr_it(image, detections, region_threshold):

    scores = detections[0]['score']

    # [x_min, y_min, width, height]
    boxes = np.array([detections[0]['bounding_box']])
 
    # Obtendo as dimensoes da imagem de entrada
    width =  image.shape[1]
    height = image.shape[0]   
    
    # Filtrando regiao de interesse e aplicando OCR
    for idx, box in enumerate(boxes):
        import ipdb; ipdb.set_trace()
        roi = box*[height, width, height, width]
        region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        return region


############### Modelo indiano

def auto_canny(image, sigma=0.33):
  # compute the median of the single channel pixel intensities
  v = np.median(image)

  # apply automatic Canny edge detection using the computed median
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edged = cv2.Canny(image, lower, upper)

  # return the edged image
  return edged


def opencvReadPlate(img):

  charList=[]
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  thresh_inv = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,39,1)
  edges = auto_canny(thresh_inv)
  ctrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
  img_area = img.shape[0]*img.shape[1]

  for i, ctr in enumerate(sorted_ctrs):
    x, y, w, h = cv2.boundingRect(ctr)
    roi_area = w*h
    non_max_sup = roi_area/img_area

    # import ipdb; ipdb.set_trace()
    if((non_max_sup >= 0.015) and (non_max_sup < 0.09)):
      # import ipdb; ipdb.set_trace()
      if ((h>1.2*w) and (3*w>=h)):
        # import ipdb; ipdb.set_trace()
        char = img[y:y+h,x:x+w]
        charList.append(cnnCharRecognition(char))
        cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)

  import ipdb; ipdb.set_trace()
  cv2.imshow('OpenCV character segmentation',img)
  licensePlate="".join(charList)
  return licensePlate


def cnnCharRecognition(img):

  dictionary = {0:'0', 1:'1', 2 :'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A',
  11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K',
  21:'L', 22:'M', 23:'N', 24:'P', 25:'Q', 26:'R', 27:'S', 28:'T', 29:'U',
  30:'V', 31:'W', 32:'X', 33:'Y', 34:'Z'}

  blackAndWhiteChar=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blackAndWhiteChar = cv2.resize(blackAndWhiteChar,(75,100))
  image = blackAndWhiteChar.reshape((1, 100,75, 1))
  image = image / 255.0
  new_predictions = characterRecognition.predict(image)
  char = np.argmax(new_predictions)
  return dictionary[char]


if __name__ == '__main__':

  # limiar de detecção e limiar de extração da licença
  detection_threshold = 0.8
  region_threshold = 0.55

  name_img = 'DSQ-6618.jpg'
  # name_img = 'Cars401.png'
  # name_img = 'BAG-7751 (3).jpg'
  # name_img = 'EUL-0433.jpg'
  # name_img = 'AXX-1773.jpg'

 
  IMAGE_PATH = os.path.join(paths['IMAGE_PATH_TEST'], name_img)

  # Realiza a identificação de placa em uma imagem
  img = cv2.imread(IMAGE_PATH)
  x_shape, y_shape, _ = img.shape

  if 'uint' in CUSTOM_MODEL_NAME:
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224,224))
    img = np.array(img)/255
    original = False
  else:
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (320,320))
    # img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224,224))
    img = np.array(img)
    original = True

  plt.imshow(img)

  interpreter = tf.lite.Interpreter(paths['TFLITE_PATH'])
  interpreter.allocate_tensors()
  
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  try:
    result = detect_objects(interpreter, img, detection_threshold, original)
  except:
    result = detect_one_object(interpreter, img, detection_threshold)

  try:
    save_image(img, result, name_img, x_shape, y_shape)
  except:
    print('não foi possível detectar!')
    exit(0)

  roi = ocr_it(img, result, region_threshold)
  name_region, format_region= name_img.split(".")
  cv2.imwrite(f'./output/{name_region}_placa.{format_region}', cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
  import ipdb; ipdb.set_trace()
  
  tflite_output = opencvReadPlate(roi)

  DEFAULT_ALPHABET = string.digits + string.ascii_lowercase
  alphabets = DEFAULT_ALPHABET
  blank_index = len(alphabets)

  final_output = "".join(alphabets[index] for index in tflite_output[0] if index not in [blank_index, -1])
  print(final_output)
  import ipdb; ipdb.set_trace()
  # print(final_output)
  # cv2.imread(image_path)




