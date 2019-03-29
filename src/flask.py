# flask_ngrok_example.py
from flask import Flask
from flask import jsonify
from flask import request
from flask import g
from flask_ngrok import run_with_ngrok

from werkzeug.serving import WSGIRequestHandler

WSGIRequestHandler.protocol_version = 'HTTP/1.1'

import PIL
from PIL import Image as PIL_Image
import ujson

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import time
import numpy as np

import zlib
import lz4
from lz4 import frame

# What model to use in GCS
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
YOUR_GCS_BUCKET = 'id-norm'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = f'gs://{YOUR_GCS_BUCKET}/exported_graphs/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = f'gs://{YOUR_GCS_BUCKET}/data/label_map.pbtxt'

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size

  # make sure we ignore Alpha layer if any
  return np.array(image.getdata()).reshape(
      (im_height, im_width, -1))[:, :, :3].astype(np.uint8)


def load_model():
    detection_graph = tf.Graph()

    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()

      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    return detection_graph, category_index


def run_inference_for_single_image(image, graph):
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
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)

            # input data feed dict is 'image_tensor:0' - this is the name of the
            # input parameter given to exporting script
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict


@app.route("/predict", methods=['POST'])
def inference():
    """
    INPUT
    { 'img': PIL image in the form of nparray }

    OUTPUT
    bboxes, classes
    """
    # data = ujson.loads(request.data)
    start = time.time()
    # data = ujson.loads(zlib.decompress(request.get_data()).decode('utf-8'))
    data = ujson.loads(lz4.frame.decompress(request.get_data()).decode('utf-8'))
    # data = ujson.loads(lz4framed.decompress(request.get_data()).decode('utf-8'))
    duration1 = time.time() - start

    bboxes, classes = np.array([]), np.array([])

    if 'img' in data:
        arr = np.array(data['img']).astype(np.uint8)
        # print(f'{arr.shape}')
        pil_im = PIL.Image.fromarray(arr)
        detection_api_input_im = load_image_into_numpy_array(pil_im)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)

        # confidence threshold is 0.8
        indic = np.argmax(output_dict['detection_scores'])
        if output_dict['detection_scores'][indic] >= 0.8:
            bboxes = output_dict['detection_boxes'][indic]
            classes = output_dict['detection_classes'][indic]

    duration2 = time.time() - start
    print(f'executime time breakdown: {round(duration1, 2)}, ' +
          f'+{round(duration2-duration1, 2)}')

    return ujson.dumps({'bboxes': bboxes.tolist(), 'classes': classes.tolist()})


@app.route("/")
def hello():
    return "Hello World!"


if __name__ == '__main__':
    app.run()
