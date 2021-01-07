#!/usr/bin/env python3

import os

import tensorflow as tf

from ../exercise_ws/src/object_detection/include/object_detection import Model
MODEL_PATH="../exercise_ws/src/object_detection/include/object_detection"

def main():
    #Load the model
    detection_model = Model()
    pass
    # TODO train loop here!
    tf.keras.backend.set_learning_phase(True)

    # These parameters can be tuned; since our training set has 5 images
    # it doesn't make sense to have a much larger batch size, though we could
    # fit more examples in memory if we wanted to.
    batch_size = 32
    learning_rate = 0.15
    num_batches = 100

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
    def train_step_fn(image_tensors,
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
            preprocessed_images = tf.concat(
                [detection_model.preprocess(image_tensor)[0]
                for image_tensor in image_tensors], axis=0)
            prediction_dict = model.predict(preprocessed_images, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
            gradients = tape.gradient(total_loss, vars_to_fine_tune)
            optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
        return total_loss

    return train_step_fn

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    train_step_fn = get_model_train_step_function(
        detection_model, optimizer, to_fine_tune)

    print('Start training!', flush=True)
    for idx in range(num_batches):
        # Grab keys for a random subset of examples
        all_keys = list(range(len(train_images_np)))
        random.shuffle(all_keys)
        example_keys = all_keys[:batch_size]

        # Note that we do not do data augmentation in this demo.  If you want a
        # a fun exercise, we recommend experimenting with random horizontal flipping
        # and random cropping :)
        gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
        gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
        image_tensors = [train_image_tensors[key] for key in example_keys]

        #print(gt_boxes_list, gt_classes_list, image_tensors)
        # Training step (forward pass + backwards pass)
        total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)

        if idx % 10 == 0:
            print('batch ' + str(idx) + ' of ' + str(num_batches)
            + ', loss=' +  str(total_loss.numpy()), flush=True)

    print('Done training!')

    # TODO don't forget to save the model's weights inside of f"{MODEL_PATH}/weights`!
    # This will be where we save checkpoint & config for TFLite conversion later.
    output_directory = os.path.join(MODEL_PATH, 'output/')
    output_checkpoint_dir = os.path.join(output_directory, 'checkpoint')
    
    # To save checkpoint for TFLite conversion.
    exported_ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt_manager = tf.train.CheckpointManager(
        exported_ckpt, output_checkpoint_dir, max_to_keep=1)
    ckpt_manager.save()
    print('Checkpoint saved!')

if __name__ == "__main__":
    main()