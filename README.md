# Custom Train Google Object Detection API to Identify ID BBox

## Google Object Detection API Config Samples
1. Github: https://github.com/hailusong/models/blob/master/research/object_detection/samples/configs
2. The one we use is **faster_rcnn_resnet101_pets.config**
    - See [this](https://gitlab.com/hailusong/openhack-ml-2018/blob/master/arctiq-ml-2018/readme.MD#the-how-to-train-in-details) about Google Object API custom training

## Information
1. god_idclass_gcs.ipynb
    - Setup GCS for Google Object Detection API custom training
2. god_idclass_colabtrain.ipynb
    - Train Google Object Detection API with custom data and pipeline configuration on CoLab
3. god_idclass_mlabtrain.ipynb
    - Train Google Object Detection API with custom data and pipeline configuration on Google Cloud ML
4. god_idclass_export.ipynb
    - Export the custom train result, a checkpoint model of Google Object Detection API, for inference
5. god_idclass_colabeval.ipynb
    - object detection inference using exported model
    - note that in **the legacy Google Object Detection API**, you need to run **train** and **eval** at the same time in separated processes
    - in the **newer Google Object Detection API** (>=1.13.1), it is one run (via **model_main/py** or TPU version) and trigger **BOTH train/eval** at the same time
    - In **tensorboard**, all **metrics** are on **Valid dataset**, **NOT** on **Train dataset**, including **IMAGEs**

### Inference with Frozen Graph
1. Load frozen Graph
    - Create Graph object
    - Create Graph Definition object
    - Load frozen graph using Graph Definition object
    - Import Graph Definition object into Graph object
2. Locate all tensors/ops we need:
    - Image input tensor/op: **image_tensor:0**
    - Output tensor/op: **detection_boxes**
    - Output tensor/op: **detection_masks**
    - Output tensor/op: **num_detections**
3. Compute the graph within **a TF session**
    - Set the **session input dict (a.k.a. feed_dict)** to the image to be inferenced
    - From the **session output dict** (a.k.a. session output) fetch values from those tensors
    - Plot the result
