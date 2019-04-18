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

import tensorflow as tf
import dlib

TF_RT_VERSION='1.13'
print(tf.__version__)
assert(tf.__version__.startswith(TF_RT_VERSION + '.')), f'tf.__version__ {tf.__version__} not matching with specified TF runtime version env variable {TF_RT_VERSION}'

# What model to use in GCS
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
YOUR_GCS_BUCKET = 'id-norm'

MODEL_NAME = 'ssd_mobilenet_v2'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = f'gs://{YOUR_GCS_BUCKET}/exported_graphs_{MODEL_NAME}/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = f'gs://{YOUR_GCS_BUCKET}/data_{MODEL_NAME}/label_map.pbtxt'

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size

  # make sure we ignore Alpha layer if any
  return np.array(image.getdata()).reshape(
      (im_height, im_width, -1))[:, :, :3].astype(np.uint8)


def load_keypoints_model()->dlib.shape_predictor:
    """
    load train Dlib key points detection model
    """
    DEFAULT_HOME='/content'
    id_keypoints_dat = f'{DEFAULT_HOME}/id_keypoints_dlib.dat'

    return dlib.shape_predictor(id_keypoints_dat)


def run_dlib_keypoints_inference(predictor:dlib.shape_predictor, im:PIL_Image, rect:list):
    np_im = np.array(im, dtype=np.uint8)
    box_left, box_top, box_width, box_height = rect
    dlib_rect = dlib.rectangle(left=box_left, top=box_top, right=box_left + box_width - 1, bottom=box_top + box_height - 1)
    shape = predictor(np_im, dlib_rect)

    return [(shape.part(part_idx).x, shape.part(part_idx).y) for part_idx in range(shape.num_parts)]


def load_model():
    """
    import Google object detection model from GCS bucket directly
    """
    detection_graph = tf.Graph()

    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()

      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    return detection_graph, category_index


# load inference model at the beginning of the app
detection_graph, _ = load_model()
dlib_shape_predictor = load_keypoints_model()


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


def _inference(pil_im:PIL.Image):
    """
    pil_im:
        a PIL.Image for key points detection
    """
    image_np = load_image_into_numpy_array(pil_im)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)

    # confidence threshold is 0.8
    indic = np.argmax(output_dict['detection_scores'])
    confidence = output_dict['detection_scores'][indic]
    if  confidence >= 0.8:
        # according to https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py
        # ymin, xmin, ymax, xmax = box
        # ------------------------------
        # bboxes = output_dict['detection_boxes'][indic]
        # classes = output_dict['detection_classes'][indic]
        pred_bbox = output_dict['detection_boxes'][indic]
        pred_bbox_xfirst = pred_bbox[[1, 0, 3, 2]].tolist()
        pred_class = output_dict['detection_classes'][indic].tolist()

        # run dlib key points detection
        # note that dlib expects the bbox to be in image coordinates, not normalize (0 to 1).
        # also we cannot assume image has the same width/height
        pred_bbox_xfirst_abs = [
            int(pred_bbox_xfirst[0] * pil_im.size[0]),
            int(pred_bbox_xfirst[1] * pil_im.size[1]),
            int(pred_bbox_xfirst[2] * pil_im.size[0]),
            int(pred_bbox_xfirst[3] * pil_im.size[1])]
        kpts = run_dlib_keypoints_inference(dlib_shape_predictor, image_np, pred_bbox_xfirst_abs)
        return confidence, pred_bbox_xfirst, pred_bbox_xfirst_abs, pred_class, kpts

    return confidence, [], [], [], []


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

    # bboxes, classes = np.array([]), np.array([])
    #
    #  |<---- image 1 ------>|, |<----- image 2 ----->|
    # bboxes:
    # [[(x1, y1, x2, y2), ...], ...]
    # classes:
    # [[class1, ...          ], [...], ...]
    # keypoints:
    # [[(x0, y0), (x, y1), ..], ...]
    bboxes, classes, keypoints = [], [], []

    if 'img' in data:
        arr = np.array(data['img']).astype(np.uint8)
        # print(f'{arr.shape}')
        pil_im = PIL.Image.fromarray(arr)

        confidence, pred_bbox_xfirst, pred_bbox_xfirst_abs, pred_class, kpts = _inference(pil_im)
        if  confidence >= 0.8:
            bboxes.append(pred_bbox_xfirst)
            classes.append(pred_class)
            keypoints.append(kpts)

    duration2 = time.time() - start
    print(f'executime time breakdown: {round(duration1, 2)}, ' +
          f'+{round(duration2-duration1, 2)}')

    return ujson.dumps({'bboxes': bboxes, 'classes': classes, 'keypoints': keypoints})


@app.route("/")
def hello():
    return "Hello World!"


if __name__ == '__main__':
    app.run()
