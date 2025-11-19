import keras_cv
import tensorflow as tf

def load_yolo():
    
    model = keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_l_backbone_coco"
    )
    return model

def run_yolo(model, image):
    
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    img = tf.expand_dims(img, 0)
    features = model(img)
    return features  # dict of P3, P4, P5