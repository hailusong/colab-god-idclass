# dlib
source: [Training alternative Dlib Shape Predictor models using Python](https://medium.com/datadriveninvestor/training-alternative-dlib-shape-predictor-models-using-python-d1d8f8bd9f5c)<br>
source: [Luca96/dlib-minified-models](https://github.com/Luca96/dlib-minified-models/tree/master/face_landmarks)<br>

## Dlib pre-trained models
Two shape predictor models (available [here](https://github.com/davisking/dlib-models)) on the iBug 300-W dataset, that respectively localize 68 (SP68) and 5 (SP5) landmark points within a face image.<br>
<img src='https://cdn-images-1.medium.com/max/1200/1*96UT-D8uSXjlnyvs9DZTog.png' width=50%/><br>
<br>
[**iBug 300W Datasets**](http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz):
1. Training
    - labels_ibug_300W_train.xml

      ```
      <?xml version='1.0' encoding='ISO-8859-1'?>
      <?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>
      <dataset>
      <name>iBUG face point dataset - training images</name>
      <comment>...</comment>
      <images>
        <image file='lfpw/trainset/image_0457.png'>
          <box top='78' left='74' width='138' height='140'>
            <part name='00' x='55' y='141'/>
            <part name='01' x='59' y='161'/>
            ...
          </box>
        </image>
        <image file='helen/trainset/2659264056_1.jpg'>
          <box top='130' left='31' width='447' height='447'>
            <part name='00' x='107' y='150'/>
            <part name='01' x='99' y='238'/>
            ...
            <part name='67' x='1440' y='1771'/>
          </box>
        </image>
      </images>
      </dataset>
      ```
2. Testing
    - labels_ibug_300W_test.xml
    - See **train.xml** above for the format

## Notes
1. Convert pnts-\*.csv and bbox-\*.csv to dlib XML format
2. Train with dlib
