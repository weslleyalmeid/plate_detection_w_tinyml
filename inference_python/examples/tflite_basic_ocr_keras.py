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
# CUSTOM_MODEL_NAME = 'detect_original.tflite'
DEFAULT_ALPHABET = string.digits + string.ascii_lowercase
# CUSTOM_MODEL_NAME = 'detect_model_int_convert.tflite'
# CUSTOM_MODEL_NAME = 'detect_model_uint8_convert.tflite'
# CUSTOM_MODEL_NAME = 'detect_model_normal_convert.tflite'
# CUSTOM_MODEL_NAME = 'detect_model_normal_convert_cli.tflite'
# CUSTOM_MODEL_NAME = 'detect_model_float16.tflite'
# CUSTOM_MODEL_NAME = 'float32.tflite'
CUSTOM_MODEL_NAME = 'float32_320_320.tflite'


paths = {
    'IMAGE_PATH_TEST': os.path.join(os.path.join(path_default, 'data'), 'test_br'),
    'OUTPUT_PATH': os.path.join(path_default, 'output'),
    'TFLITE_PATH': os.path.join(path_default, 'models', CUSTOM_MODEL_NAME),
}


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


def detect_objects(interpreter, image, threshold, my_func=False):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    
    if my_func:
        import ipdb; ipdb.set_trace()
        # Get all output details
        boxes = get_output_tensor(interpreter, 1)
        classes = get_output_tensor(interpreter, 3)
        scores = get_output_tensor(interpreter, 0)
        count = int(get_output_tensor(interpreter, 2))
        import ipdb; ipdb.set_trace()

    else:
        # Get all output details
        boxes = get_output_tensor(interpreter, 0)
        classes = get_output_tensor(interpreter, 1)
        scores = get_output_tensor(interpreter, 2)

        # TODO: quando rodar em realtime, lembrar de alterar esse cara para o comentado
        count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            import ipdb; ipdb.set_trace()
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def save_image(img, result, name_img, x_shape, y_shape, model_x_shape, model_y_shape):
    import ipdb; ipdb.set_trace()
    CAMERA_WIDTH = model_x_shape
    CAMERA_HEIGHT = model_y_shape
    ymin, xmin, ymax, xmax = result[0]['bounding_box']
    import ipdb; ipdb.set_trace()
    xmin = int(max(1, xmin * CAMERA_WIDTH))
    xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
    ymin = int(max(1, ymin * CAMERA_HEIGHT))
    ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
    import ipdb; ipdb.set_trace()
    rect = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    print(xmin, xmax, ymin, ymax)
    plt.imshow(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)); plt.show()
    cv2.imwrite(os.path.join(paths['OUTPUT_PATH'], name_img), cv2.resize(
        cv2.cvtColor(rect, cv2.COLOR_BGR2RGB), (x_shape, y_shape)))


def run_tflite_model_ocr(image, quantization):
    # https://colab.research.google.com/github/tulasiram58827/ocr_tflite/blob/main/colabs/KERAS_OCR_TFLITE.ipynb#scrollTo=rkGs7BDWf6dm
    # TODO: se ler a imagem já salva
    # input_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # input_data = cv2.resize(input_data, (200, 31))
    # input_data = input_data[np.newaxis]
    # input_data = np.expand_dims(input_data, 3)
    # input_data = input_data.astype('float32')/255

    # TODO: recebendo roi
    input_data = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    input_data = cv2.resize(image, (200, 31))
    input_data = (input_data-255)/255
    input_data = input_data.astype('float32')
    input_data = input_data[:, :, :1]
    input_data = input_data[np.newaxis]
    # 'shape': array([  1,  31, 200,   1]

    # plt.imshow(image); plt.show()
    # plt.imshow(np.where(input_data > 60, 255, 0)); plt.show()

    import ipdb
    ipdb.set_trace()

    path = f'./models/lite-model_keras-ocr_{quantization}.tflite'
    interpreter_ocr = tf.lite.Interpreter(model_path=path)
    interpreter_ocr.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter_ocr.get_input_details()
    output_details = interpreter_ocr.get_output_details()

    input_shape = input_details[0]['shape']
    interpreter_ocr.set_tensor(input_details[0]['index'], input_data)

    interpreter_ocr.invoke()

    output = interpreter_ocr.get_tensor(output_details[0]['index'])
    import ipdb
    ipdb.set_trace()

    get_plate(output)
    return output


def get_roi(image, detections, region_threshold):

    scores = detections[0]['score']

    # [x_min, y_min, width, height]
    boxes = np.array([detections[0]['bounding_box']])

    # Obtendo as dimensoes da imagem de entrada
    width = image.shape[1]
    height = image.shape[0]

    # Filtrando regiao de interesse e aplicando OCR
    for idx, box in enumerate(boxes):
        import ipdb; ipdb.set_trace()
        roi = box*[height, width, height, width]
        region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]
        return region


def get_plate(tflite_output):
    alphabets = DEFAULT_ALPHABET
    blank_index = len(alphabets)

    final_output = "".join(
        alphabets[index] for index in tflite_output[0] if index not in [blank_index, -1])

    return final_output


if __name__ == '__main__':

    # limiar de detecção e limiar de extração da licença
    detection_threshold = 0.6
    region_threshold = 0.55

    # name_img = 'EUL-0433.jpg'
    # name_img = 'DSQ-6618.jpg'
    # name_img = 'Cars401.png'
    # name_img = 'BAG-7751.jpg'
    name_img = 'EJD-3779.jpg'
    

    IMAGE_PATH = os.path.join(paths['IMAGE_PATH_TEST'], name_img)

    # Realiza a identificação de placa em uma imagem
    img = cv2.imread(IMAGE_PATH)

    x_shape, y_shape, _ = img.shape
    interpreter = tf.lite.Interpreter(paths['TFLITE_PATH'])
    interpreter.allocate_tensors()

    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    if 'uint' in CUSTOM_MODEL_NAME:
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224))
        img = np.array(img)/255
    else:
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (input_width, input_height))
        # img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (640, 640))
        # img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (640, 640))
        img = np.array(img)

    plt.imshow(img)
    
    result = detect_objects(interpreter, img, detection_threshold, my_func=True)

    try:
        save_image(img, result, name_img, x_shape, y_shape, input_width, input_height)
    except:
        print('não foi possível detectar!')
        # continue

    roi = get_roi(img, result, region_threshold)
    name_region, format_region = name_img.split(".")
    # cv2.imwrite(f'./output/{name_region}_placa.{format_region}',
    #             cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    import ipdb
    ipdb.set_trace()

    # tflite_output = run_tflite_model_ocr(roi, 'float16')
    tflite_output = run_tflite_model_ocr(roi, 'dr')

    plate_output = get_plate(tflite_output)
    print(plate_output)
    import ipdb
    ipdb.set_trace()
    # print(final_output)
    # cv2.imread(image_path)
