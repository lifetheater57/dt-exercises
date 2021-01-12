#!/usr/bin/env python3

import os
from shutil import copy
from datetime import datetime
import argparse
import numpy as np

import tensorflow as tf

import inspect
import sys
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from exercise_ws.src.object_detection.include.object_detection.model import Model
from exercise_ws.src.object_detection.include.object_detection.dataset import Dataset

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
    
    # Be sure to be in this file's directory
    os.chdir(current_dir)

    tf.keras.backend.clear_session()

    # Preparing the dataset
    dataset = Dataset()
    
    detection_model = Model(dataset.num_classes).detection_model

    # This will be where we save checkpoint & config for TFLite conversion later.
    output_directory = os.path.join(MODEL_PATH, 'output/')
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
    num_epochs = 1

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

    ragged_batches = dataset.frames.apply(
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
            
            # Training step (forward pass + backward pass)
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
    print("Start converting model to TF Lite.")
    
    os.chdir(MODEL_PATH)
    
    # Generate a TFLite-friendly intermediate SavedModel.
    os.system(f"python models/export_tflite_graph_tf2.py \
                --pipeline_config_path pipeline.config \
                --trained_checkpoint_dir output/checkpoint/ \
                --output_directory tflite")

    # Generate the final model from the intermediate model using TensorFlow Lite Converter. 
    os.system(f"tflite_convert \
                --saved_model_dir=tflite/saved_model \
                --output_file=tflite/model.tflite")

    copy("tflite/model.tflite", os.path.join(
        last_checkpoint_dir.replace(MODEL_PATH + "/", ""), "model.tflite")
    )
    copy("tflite/model.tflite", os.path.join(
        output_checkpoint_dir.replace(MODEL_PATH + "/", ""), "model.tflite")
    )

    print("Done converting model to TF Lite.")

if __name__ == "__main__":
    main()
    