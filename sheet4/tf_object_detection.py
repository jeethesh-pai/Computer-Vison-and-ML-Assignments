# Task 4
#
# Load and use a pretrained `Object Detection` model from the `TensorFlow Model Garden`

# Download/cache required model and test data
import tensorflow as tf
import os
import time
from utils import *
from tf_utils import *
import cv2
import numpy as np
from models.research.object_detection.utils import visualization_utils as viz_utils
import matplotlib
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util

tf.get_logger().setLevel('ERROR')

model_name = input('model name: 1 - centernet Hourglass, 2 - centernetResnet50 ')
print('Download model...')
if int(model_name) == 1:
    modelPath = download_model('20200713', 'centernet_hg104_1024x1024_coco17_tpu-32')
    print(modelPath)
else:
    modelPath = download_model('20200711', 'centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8')
    print(modelPath)

print('Download labels...')
labelsPath = download_labels('mscoco_label_map.pbtxt')
print(labelsPath)

print('Download test images...')
imagePaths = download_test_images(['image1.jpg', 'image2.jpg'])
print('/n'.join(imagePaths))

# Load the model

print('Loading model...')
savedModelPath = os.path.join(modelPath, "saved_model")
start_time = time.time()
model = tf.saved_model.load(savedModelPath)
end_time = time.time()
print('Time taken to load the model:%.2f secs' % (end_time - start_time))

# Loading label_map data
category_index = label_map_util.create_category_index_from_labelmap(labelsPath, use_display_name=True)

# Run inference

matplotlib.use('TkAgg')  # Reactivate GUI backend (deactivated by `import viz_utils`)

imgs = []
for image_path in imagePaths:
    print('Running inference for {}... '.format(image_path))
    image = cv2.imread(image_path)
    img = tf.convert_to_tensor(image, dtype=tf.uint8)
    img = tf.expand_dims(img, axis=0)
    start_predict_time = time.time()
    detections = model(img)
    end_predict_time = time.time()
    print('Time taken for prediction: %.2f sec' % (end_predict_time - start_predict_time))

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    img = tf.squeeze(img)

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)
    imgs.append(image)
plt.figure(figsize=(13, 5))
showImages(imgs)
