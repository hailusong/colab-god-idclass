# get the option object
options = dlib.shape_predictor_training_options()

# set the parameters
# model capacity: 4 layer is optimal value
options.tree_depth = 4

# regularization parameter
# the ability of the model to generalize and learn patterns instead of fixed-data
# -> 1: over-fitting
# -> 0: risk of under-fitting, need lost of data (thousands)
options.nu = 0.1

# cascade_depth affects either the size and accuracy of a model
# 15: a perfect balance of maximum accuracy and a reasonable model-size
options.cascade_depth = 10

# the number of pixels used to generate the features for the random trees at each cascade
# 400: a great accuracy with a good runtime speed
# 800 or 1k: superior precision but very slow
# 100 to 150: reasonable good accuracy but with an impressing runtime speed
options.feature_pool_size = 400

# number of split features sampled at each node
# affects the training speed and the model accuracy
# higher number (upto 100 or 300): increase accuracy but not size increasing too much
options.num_test_splits = 20

# the number of randomly selected deformations applied to the training samples
# Applying random deformations to the training images is a simple technique that effectively increase the size of the training dataset
# small dataset: use the value 20 to 40
options.oversampling_amount = 20

# available in the latest dlib release
# make the model more robust against eventually misplaced face regions
options.oversampling_translation_jitter = 0
