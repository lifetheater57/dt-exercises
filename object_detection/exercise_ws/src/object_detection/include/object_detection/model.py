import subprocess
import os

import tensorflow as tf

from object_detection.utils import config_util
from object_detection.builders import model_builder

class NoGPUAvailable(Exception):
    pass

class Wrapper():
    def __init__(self, model_file):
        # TODO Instantiate your model and other class instances here!
        # TODO Don't forget to set your model in evaluation/testing/production mode, and sending it to the GPU
        
        # TODO If no GPU is available, raise the NoGPUAvailable exception
        if len(tf.config.list_physical_devices("GPU")) == 0:
            raise NoGPUAvailable()

    def predict(self, batch_or_image):
        # TODO Make your model predict here!

        # TODO The given batch_or_image parameter will be a numpy array (ie either a 224 x 224 x 3 image, or a
        # TODO batch_size x 224 x 224 x 3 batch of images)
        # TODO These images will be 224 x 224, but feel free to have any model, so long as it can handle these
        # TODO dimensions. You could resize the images before predicting, make your model dimension-agnostic somehow,
        # TODO etc.

        # TODO This method should return a tuple of three lists of numpy arrays. The first is the bounding boxes, the
        # TODO second is the corresponding labels, the third is the scores (the probabilities)

        # See this pseudocode for inspiration
        boxes = []
        labels = []
        scores = []
        for img in batch_or_image:  # or simply pipe the whole batch to the model instead of using a loop!

            box, label, score = self.model.predict(img) # TODO you probably need to send the image to a tensor, etc.
            boxes.append(box)
            labels.append(label)
            scores.append(score)

        return boxes, labels, scores
                
        
        shape = batch_or_image.shape
        if len(shape) == 3:
            batch_or_image = batch_or_image.reshape(
                (1, shape[0], shape[2], shape[3])
            )        
        
        preprocessed_images = tf.concat(
            [model.preprocess(batch_or_image[i])[0]
            for i in range(batch_or_image.shape[0])], axis=0
        )
            
        shapes = tf.constant(batch_or_image.shape[0] * [[320, 320, 3]], dtype=tf.int32)

        prediction_dict = model.predict(preprocessed_images, shapes)
        predictions = model.postprocess(prediction_dict, shapes)
        

class Model():    # TODO probably extend a TF or Pytorch class!
    def __init__(self):
        pass
        # TODO Instantiate your weights etc here!
        subprocess.call("bash TF-Lite_model.sh", shell=True)

        tf.keras.backend.clear_session()

        print('Building model and restoring weights for fine-tuning...', flush=True)
        num_classes = 4
        pipeline_config = 'models/research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config'
        checkpoint_path = 'models/research/object_detection/test_data/checkpoint/ckpt-0'

        # This will be where we save checkpoint & config for TFLite conversion later.
        output_directory = 'output/'

        # Load pipeline config and build a detection model.
        #
        # Since we are working off of a COCO architecture which predicts 90
        # class slots by default, we override the `num_classes` field here to be just
        # four (the number of classes in the dataset).
        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        model_config = configs['model']
        model_config.ssd.num_classes = num_classes
        model_config.ssd.freeze_batchnorm = True
        detection_model = model_builder.build(
            model_config=model_config, is_training=True)
        # Save new pipeline config
        pipeline_proto = config_util.create_pipeline_proto_from_configs(configs)
        config_util.save_pipeline_config(pipeline_proto, output_directory)

        # Set up object-based checkpoint restore --- SSD has two prediction
        # `heads` --- one for classification, the other for box regression.  We will
        # restore the box regression head but initialize the classification head
        # from scratch (we show the omission below by commenting out the line that
        # we would add if we wanted to restore both heads)
        box_predictor = tf.compat.v2.train.Checkpoint(
            _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
            # _prediction_heads=detection_model._box_predictor._prediction_heads,
            #    (i.e., the classification head that we *will not* restore)
            _box_prediction_head=detection_model._box_predictor._box_prediction_head,
            )
        model = tf.compat.v2.train.Checkpoint(
                _feature_extractor=detection_model._feature_extractor,
                _box_predictor=box_predictor)
        ckpt = tf.compat.v2.train.Checkpoint(model=model)
        ckpt.restore(checkpoint_path).expect_partial()

        # Run model through a dummy image so that variables are created
        image, shapes = detection_model.preprocess(tf.zeros([1, 320, 320, 3]))
        prediction_dict = detection_model.predict(image, shapes)
        _ = detection_model.postprocess(prediction_dict, shapes)
        print('Weights restored!')

        self = detection_model
    # TODO add your own functions if need be!
