import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
import time

# tamvm
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def create_graph():
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = './frozen_inference_graph.pb'
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def run_inference_for_single_image(image):
  graph = create_graph()
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def tflite_run_inference_for_single_image(image):
    # Load TFLite model and allocate tensors.
    # model_path = '/home/tamvm/AndroidStudioProjects/MLKitDemo/app/src/main/assets/detect_face_224_quantized.tflite'
    model_path = '/home/tamvm/Projects/tensorflow-models/research/object_detection/face112x112.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('input details = ', input_details)
    print('output details = ', input_details)
    # Test model on random input data.
    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_data = image
    print('required shape = ', input_shape, ' actual = ', input_data.shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    t0 = time.clock()
    interpreter.invoke()
    t1 = time.clock() - t0
    print('inference time = ', t1)
    # TFLite_Detection_PostProcess custom op node has four outputs:
    # detection_boxes: a float32 tensor of shape [1, num_boxes, 4] with box
    # locations
    # detection_classes: a float32 tensor of shape [1, num_boxes]
    # with class indices
    # detection_scores: a float32 tensor of shape [1, num_boxes]
    # with class scores
    # num_boxes: a float32 tensor of size 1 containing the number of detected boxes
    output_dict = {
        'detection_boxes': interpreter.get_tensor(output_details[0]['index'])[0],
        'detection_classes': interpreter.get_tensor(output_details[1]['index'])[0],
        'detection_scores': interpreter.get_tensor(output_details[2]['index'])[0],
        'num_boxes': interpreter.get_tensor(output_details[3]['index'])[0]
    }
    # print(output_dict)
    print('no of detections = ', output_dict['num_boxes'])
    output_dict['detection_classes'] = [int(x) for x in output_dict['detection_classes']]
    return output_dict


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'face_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

PATH_TO_TEST_IMAGES_DIR = './WIDER_train/images/0--Parade'
# TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, '0_Parade_Parade_0_{}.jpg'.format(i)) for i in [924, 950]]
TEST_IMAGE_PATHS = ['/home/tamvm/Downloads/test_face_detect.jpg', 
                    '/home/tamvm/Pictures/device-2019-05-07-185923.jpg',
                    '/home/tamvm/Pictures/test_face_detect_2.jpeg']
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

for image_path in TEST_IMAGE_PATHS:
    print('Process image ', image_path)
  # Actual detection.
    use_lite_model = True
    if use_lite_model:
        crop_size = 112
        image = Image.open(image_path)
        image = image.resize((crop_size, crop_size), Image.ANTIALIAS)
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = tflite_run_inference_for_single_image(image_np_expanded)
        print('output_dict', output_dict)
    else:
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = run_inference_for_single_image(image_np_expanded)
        # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.show()