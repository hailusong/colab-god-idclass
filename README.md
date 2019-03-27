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
