#!/usr/bin/env python3

import os
from datetime import datetime
import argparse
import numpy as np

import tensorflow as tf
tfds = tf.data.Dataset

import sys
sys.path.insert(0, "..")
from exercise_ws.src.object_detection.include.object_detection.model import Model
MODEL_PATH="../exercise_ws/src/object_detection/include/object_detection"

# To prevent the process to use all the VRAM if not necessary
# Also, allow to use a RTX NVIDIA GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

# Most of the current script is an adaptation from this colab tutorial
# https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tflite.ipynb

def prepareDataset(file_path_pattern="../dataset/*.npz"):
    # By convention, our non-background classes start counting at 1. Given
    # that we will be predicting just one class, we will therefore assign it a
    # `class id` of 1.
    class_id = {
        "duckie" : 1,
        "cone" : 2,
        "truck" : 3,
        "bus" : 4,
    }

    category_index = {
        class_id["duckie"] : {
            "id" : class_id["duckie"],
            "name" : "duckie",
        },
        class_id["cone"] : {
            "id" : class_id["cone"],
            "name" : "cone",
        },
        class_id["truck"] : {
            "id" : class_id["truck"],
            "name" : "truck",
        },
        class_id["bus"] : {
            "id" : class_id["bus"],
            "name" : "bus",
        },
    }

    num_classes = len(class_id)

    # Convert class labels to one-hot; convert everything to tensors.
    # The `label_id_offset` here shifts all classes by a certain number of indices;
    # we do this here so that the model receives one-hot labels where non-background
    # classes start counting at the zeroth index.  This is ordinarily just handled
    # automatically in our training binaries, but we need to reproduce it here.
    label_id_offset = 1

    def prepare_frames(filename_tensor):
        filename = bytes.decode(filename_tensor.numpy())
        example = np.load(filename)

        train_image_np = example["arr_0"]
        boxes_np = example["arr_1"]
        class_np = example["arr_2"]

        image_tensor = tf.convert_to_tensor(train_image_np, dtype=tf.float32)

        boxes_np = boxes_np[:, [1, 0, 3, 2]]
        boxes_np[:, [0, 2]] /= train_image_np.shape[-3]
        boxes_np[:, [1, 3]] /= train_image_np.shape[-2]
        boxes_tensor = tf.convert_to_tensor(boxes_np, dtype=tf.float32)

        zero_indexed_groundtruth_classes = tf.convert_to_tensor(
            class_np.squeeze(axis=1), dtype=tf.int32)
        zero_indexed_groundtruth_classes -= label_id_offset
        
        gt_classes_one_hot_tensor = tf.one_hot(
            zero_indexed_groundtruth_classes, num_classes)
        
        return image_tensor, boxes_tensor, gt_classes_one_hot_tensor

    def set_shapes(image, boxes, classes):
        image.set_shape((None, None, 3))
        boxes.set_shape((None, 4))
        classes.set_shape((None, num_classes))
        
        return image, boxes, classes

    file_list = tfds.list_files(file_path_pattern)
    frame_ds = file_list.map(lambda x: tf.py_function(prepare_frames, [x], [tf.float32, tf.float32, tf.float32]))
    frame_ds = frame_ds.map(set_shapes)
    frame_ds = frame_ds.shuffle(1000, reshuffle_each_iteration=True)

    print('Done prepping data.')

    return frame_ds, class_id, category_index

def main():
    # TODO train loop here!
    
    parser = argparse.ArgumentParser(description="train object detection model")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None, 
        help="model checkpoint from which to start the training"
    )
    
    args = parser.parse_args()
    
    tf.keras.backend.clear_session()

    # Preparing the dataset
    frame_ds, class_id, category_index = prepareDataset()
    num_classes = len(class_id)
    
    detection_model = Model(num_classes).detection_model

    # This will be where we save checkpoint & config for TFLite conversion later.
    output_directory = 'output/'
    last_checkpoint_dir = os.path.join(output_directory, 'checkpoint')
    output_checkpoint_dir = os.path.join(output_directory, 'checkpoint_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    best_checkpoint_dir = os.path.join(output_checkpoint_dir, 'best_checkpoint_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    # To save checkpoint for TFLite conversion.
    exported_last_ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    last_ckpt_manager = tf.train.CheckpointManager(
        exported_last_ckpt, last_checkpoint_dir, max_to_keep=1
    )

    exported_ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt_manager = tf.train.CheckpointManager(
        exported_ckpt, output_checkpoint_dir, max_to_keep=1
    )

    exported_best_ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    best_ckpt_manager = tf.train.CheckpointManager(
        exported_best_ckpt, best_checkpoint_dir, max_to_keep=1
    )

    tf.keras.backend.set_learning_phase(True)

    # Define training config
    batch_size = 16
    learning_rate = 0.15
    num_epochs = 0

    # Restore previous training if required
    if args.checkpoint is not None:
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(args.checkpoint)

    # Select variables in top layers to fine-tune.
    trainable_variables = detection_model.trainable_variables
    to_fine_tune = []
    prefixes_to_train = [
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
    for var in trainable_variables:
        if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
            to_fine_tune.append(var)

    # Set up forward + backward pass for a single train step.
    def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
        """Get a tf.function for training step."""

        # Use tf.function for a bit of speed.
        # Comment out the tf.function decorator if you want the inside of the
        # function to run eagerly.
        #@tf.function
        def train_step_fn(preprocesed_images,
                            groundtruth_boxes_list,
                            groundtruth_classes_list):
            """A single training iteration.

            Args:
            image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
                Note that the height and width can vary across images, as they are
                reshaped within this function to be 320x320.
            groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
                tf.float32 representing groundtruth boxes for each image in the batch.
            groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
                with type tf.float32 representing groundtruth boxes for each image in
                the batch.

            Returns:
            A scalar tensor representing the total loss for the input batch.
            """
            shapes = tf.constant(batch_size * [[320, 320, 3]], dtype=tf.int32)
            model.provide_groundtruth(
                groundtruth_boxes_list=groundtruth_boxes_list,
                groundtruth_classes_list=groundtruth_classes_list)
            with tf.GradientTape() as tape:      
                prediction_dict = model.predict(preprocessed_images, shapes)
                losses_dict = model.loss(prediction_dict, shapes)
                total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
                gradients = tape.gradient(total_loss, vars_to_fine_tune)
                optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
            return total_loss

        return train_step_fn

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    train_step_fn = get_model_train_step_function(
        detection_model, optimizer, to_fine_tune
    )

    print('Start fine-tuning!', flush=True)

    ragged_batches = frame_ds.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size)
    )

    best_loss = 1e10
    step = 0
    for idx in range(num_epochs):
        for batch in ragged_batches:
            images, gt_boxes, gt_classes = batch

            preprocessed_images, _ = detection_model.preprocess(images.to_tensor())
            gt_boxes_list = [tensor for tensor in gt_boxes]
            gt_classes_list = [tensor for tensor in gt_classes]
            
            # Training step (forward pass + backwards pass)
            total_loss = train_step_fn(preprocessed_images, gt_boxes_list, gt_classes_list)

            if step % 10 == 0:
                print("\tstep", str(step) + ", loss=", total_loss.numpy(), flush=True)

            if total_loss.numpy() < best_loss:
                best_ckpt_manager.save()
                print("\tBest checkpoint saved at step", step, ", loss =", total_loss.numpy())
                best_loss = total_loss.numpy()
            
            step += 1
        
        print("batch", idx + 1, "of", num_epochs, ", loss =", total_loss.numpy(), flush=True)
    
    print('Done fine-tuning!')

    last_ckpt_manager.save()
    ckpt_manager.save()
    
    print('Checkpoint saved!')

if __name__ == "__main__":
    main()
    
    # Generate a TFLite-friendly intermediate SavedModel.
    os.system(f"models/research/object_detection/export_tflite_graph_tf2.py \
                --pipeline_config_path {MODEL_PATH}/pipeline.config \
                --trained_checkpoint_dir output/checkpoint/ \
                --output_directory {MODEL_PATH}/tflite")

    # Generate the final model from the intermediate model using TensorFlow Lite Converter. 
    os.system(f"tflite_convert \
                --saved_model_dir={MODEL_PATH}/tflite/saved_model \
                --output_file={MODEL_PATH}/tflite/model.tflite")