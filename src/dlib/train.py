import cv2
import dlib

from xml_slicing.py import slice_xml

# define the landmark-indices we're interested to localize:
# for example if we want detect the left and right eye landmarks
EYES = [i for i in range(36, 48)]
# EYES = [i for i in range(0, 4)]

def train_model(name, xml):
  '''
  requires: the model name, and the path to the xml annotations.
  It trains and saves a new model according to the specified
  training options and given annotations

  example @ https://github.com/Luca96/dlib-minified-models/tree/master/face_landmarks:
    options = dlib.shape_predictor_training_options()
    options.tree_depth = 4
    options.nu = 0.1
    options.cascade_depth = 15
    options.feature_pool_size = 800  # or even 1000
    options.num_test_splits = 200  # 150-200 is enough
  '''
  # get the training options
  options = dlib.shape_predictor_training_options()
  options.tree_depth = 4
  options.nu = 0.1
  options.cascade_depth = 15
  options.feature_pool_size = 400
  options.num_test_splits = 50
  options.oversampling_amount = 5
  #
  options.be_verbose = True  # tells what is happening during the training
  options.num_threads = 4    # number of the threads used to train the model

  # finally, train the model
  dlib.train_shape_predictor(xml, name, options)


def measure_model_error(model, xml_annotations):
    '''requires: the model and xml path.
    It measures the error of the model on the given
    xml file of annotations.'''
    error = dlib.test_shape_predictor(xml_annotations, model)
    print("Error of the model: {} is {}".format(model, error))


if __name__ == '__main__':
  # train a new model with a subset of the ibug annotations
  ibug_xml = "labels_ibug_300W_train.xml"
  eyes_xml = "eyes.xml"
  eyes_dat = "eyes.dat"

  # create the training xml for the new model with only the desired points.
  # Specify the points you require when you call the slice_xml function if you want.
  # slice_xml(ibug_xml, "four_landmark.xml", parts=[8, 30, 36, 54])
  slice_xml(ibug_xml, eyes_xml, parts=EYES)

  # finally train the eye model
  train_model(eyes_dat, eyes_xml)

  # ..and measure the model error on the testing annotations
  measure_model_error(eyes_dat, "labels_ibug_300W_test.xml")
