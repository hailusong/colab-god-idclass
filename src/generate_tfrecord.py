"""
This file is based on https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.pyself.
It is modified to generate TF Records from a slightly different CSV file.

Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        print(f'removing attribute {keys}')
        FLAGS.__delattr__(keys)


# if running inside IPython notebook, the python session will be maintained across
# cells, so does the tf.app.flags. That will cause flags defined twice error
# if we %run the app multiple times. The workaroud is to always clean up
# the flags before defining them.
flags = tf.app.flags
del_all_flags(flags.FLAGS)
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS

# this should be synced with the label_map.pbtxt
LABEL_MAP = {
    "BACKGROUND": 0,
    "ON-DL": 1,
    "ON-HC": 2,
    "UNKNOWN": 3
}


# TO-DO replace this with label map
def class_text_to_int(row_label):
    return LABEL_MAP[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    valid_range_min = -0.1
    valid_range_max = 1.1

    for index, row in group.object.iterrows():
        x1 = row['bbox1_x1'] / width
        x2 = row['bbox1_x2'] / width
        y1 = row['bbox1_y1'] / height
        y2 = row['bbox1_y2'] / height
        valid_rec = (valid_range_min < x1 <  valid_range_max and
                     valid_range_min < x2 <  valid_range_max and
                     valid_range_min < y1 <  valid_range_max and
                     valid_range_min < y2 <  valid_range_max)
        if not valid_rec:
            print(f'{x1}, {y1} - {x2}, {y2} is not completely valid bbox. Ignored')
            continue

        xmins.append(row['bbox1_x1'] / width)
        xmaxs.append(row['bbox1_x2'] / width)
        ymins.append(row['bbox1_y1'] / height)
        ymaxs.append(row['bbox1_y2'] / height)
        classes_text.append(row['label'].encode('utf8'))
        classes.append(class_text_to_int(row['label']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    print(f'total number of rows read is {examples.shape[0]}')
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
