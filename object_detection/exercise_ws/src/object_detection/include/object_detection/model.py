import os
import sys
import inspect
import subprocess

import tensorflow as tf

from object_detection.utils import config_util
from object_detection.builders import model_builder

# To prevent the process to use all the VRAM if not necessary
# Also, allow to use a RTX NVIDIA GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

class NoGPUAvailable(Exception):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
    #print("NoGPUAvailable Exception: No gpu was found, the cpu will be used.")

class ModelNotTrained(RuntimeError):
    pass
    #print("ModelNotTrained Exception: Prediction was attempted before training the model.")
    #sys.exit(1)

class Wrapper():
    def __init__(self, model_file=None):
        # TODO If no GPU is available, raise the NoGPUAvailable exception
        if len(tf.config.list_physical_devices("GPU")) == 0:
            raise NoGPUAvailable()

        # TODO Instantiate your model and other class instances here!
        # TODO Don't forget to set your model in evaluation/testing/production mode, and sending it to the GPU
        self.model = Model(4)

        if model_file is not None:
            if os.path.isfile(model_file):
                self.model.tf_lite_model_path = model_file
            else:
                print("WARNING: \"" + model_file + "\" does not exists.")

    def predict(self, batch_or_image):
        # TODO Make your model predict here!

        # TODO The given batch_or_image parameter will be a numpy array (ie either a 224 x 224 x 3 image, or a
        # TODO batch_size x 224 x 224 x 3 batch of images)
        # TODO These images will be 224 x 224, but feel free to have any model, so long as it can handle these
        # TODO dimensions. You could resize the images before predicting, make your model dimension-agnostic somehow,
        # TODO etc.

        # TODO This method should return a tuple of three lists of numpy arrays. The first is the bounding boxes, the
        # TODO second is the corresponding labels, the third is the scores (the probabilities)
        
        # Note that if using the @tf.function decorator the first frame will
        # trigger tracing of the tf.function, which will take some time, after 
        # which inference should be fast.

        input_tensor = tf.convert_to_tensor(batch_or_image, dtype=tf.float32)
        boxes, labels, scores = self.model.detect(input_tensor)
        return boxes[0], labels[0], scores[0]

class Model():    # TODO probably extend a TF or Pytorch class!
    def __init__(self, num_classes):
        # TODO Instantiate your weights etc here!
        # Set the working directory to the current file's directory
        previous_dir = os.getcwd()
        current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        os.chdir(current_dir)
        
        self.tf_lite_model_path = os.path.join(current_dir, "tflite/model.tflite")
        
        # Fetch the base model and prepares it.
        subprocess.call(f"bash ./fetch_model.sh", shell=True)
        
        tf.keras.backend.clear_session()

        print('Building model and restoring weights for fine-tuning...', flush=True)
        
        # Define config variables
        pipeline_config = 'models/configs/tf2/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config'
        checkpoint_path = 'models/test_data/checkpoint/ckpt-0'

        # Load pipeline config and build a detection model.
        #
        # Since we are working off of a COCO architecture which predicts 90
        # class slots by default, we override the `num_classes` field here to
        # be just one (for our new rubber ducky class).
        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        model_config = configs['model']
        model_config.ssd.num_classes = num_classes
        model_config.ssd.freeze_batchnorm = True
        detection_model = model_builder.build(
            model_config=model_config, is_training=True
        )
        
        # Save new pipeline config
        pipeline_proto = config_util.create_pipeline_proto_from_configs(configs)
        config_util.save_pipeline_config(pipeline_proto, ".")

        # Set up object-based checkpoint restore --- SSD has two prediction
        # `heads` --- one for classification, the other for box regression. We
        # will restore the box regression head but initialize the classification
        # head from scratch (we show the omission below by commenting out the
        # line that we would add if we wanted to restore both heads)
        fake_box_predictor = tf.compat.v2.train.Checkpoint(
            _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
            # _prediction_heads=detection_model._box_predictor._prediction_heads,
            #    (i.e., the classification head that we *will not* restore)
            _box_prediction_head=detection_model._box_predictor._box_prediction_head,
        )
        fake_model = tf.compat.v2.train.Checkpoint(
            _feature_extractor=detection_model._feature_extractor,
            _box_predictor=fake_box_predictor
        )
        ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
        ckpt.restore(checkpoint_path).expect_partial()

        # Run model through a dummy image so that variables are created
        image, shapes = detection_model.preprocess(tf.zeros([1, 320, 320, 3]))
        prediction_dict = detection_model.predict(image, shapes)
        _ = detection_model.postprocess(prediction_dict, shapes)
        
        print('Weights restored!')

        self.detection_model = detection_model
        
        # Initialize the interpreter variable
        self.interpreter = None

        # Set the working directory to the previous one
        os.chdir(previous_dir)

    # TODO add your own functions if need be!
    # Again, uncomment this decorator if you want to run inference eagerly
    #@tf.function
    def detect(self, input_tensor):
        """Run detection on an input image.

        Args:
            interpreter: tf.lite.Interpreter
            input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
            Note that height and width can be anything since the image will be
            immediately resized according to the needs of the model within this
            function.

        Returns:
            A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
            and `detection_scores`).
        """
        if not os.path.isfile(self.tf_lite_model_path):
            raise ModelNotTrained()

        if self.interpreter is None:
            # Load the TFLite model and allocate tensors.
            self.interpreter = tf.lite.Interpreter(model_path=self.tf_lite_model_path)
            self.interpreter.allocate_tensors()
        
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # We use the original model for pre-processing,
        # since the TFLite model doesn't include pre-processing.            
        preprocessed_image, shapes = self.detection_model.preprocess(input_tensor)
        
        self.interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(output_details[0]['index'])
        classes = self.interpreter.get_tensor(output_details[1]['index'])
        scores = self.interpreter.get_tensor(output_details[2]['index'])
        
        return boxes, classes, scores