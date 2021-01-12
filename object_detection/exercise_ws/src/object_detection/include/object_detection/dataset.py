import numpy as np

import tensorflow as tf
tfds = tf.data.Dataset

# To prevent the process to use all the VRAM if not necessary
# Also, allow to use a RTX NVIDIA GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

class Dataset():
    def __init__(self, file_path_pattern="../dataset/*.npz"):
        # By convention, our non-background classes start counting at 1. Given
        # that we will be predicting just one class, we will therefore assign it a
        # `class id` of 1.
        self.class_id = {
            "duckie" : 1,
            "cone" : 2,
            "truck" : 3,
            "bus" : 4,
        }

        self.category_index = {
            self.class_id["duckie"] : {
                "id" : self.class_id["duckie"],
                "name" : "duckie",
            },
            self.class_id["cone"] : {
                "id" : self.class_id["cone"],
                "name" : "cone",
            },
            self.class_id["truck"] : {
                "id" : self.class_id["truck"],
                "name" : "truck",
            },
            self.class_id["bus"] : {
                "id" : self.class_id["bus"],
                "name" : "bus",
            },
        }

        self.num_classes = len(self.class_id)

        # Convert class labels to one-hot; convert everything to tensors.
        # The `label_id_offset` here shifts all classes by a certain number of indices;
        # we do this here so that the model receives one-hot labels where non-background
        # classes start counting at the zeroth index.  This is ordinarily just handled
        # automatically in our training binaries, but we need to reproduce it here.
        self.label_id_offset = 1

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
            zero_indexed_groundtruth_classes -= self.label_id_offset
            
            gt_classes_one_hot_tensor = tf.one_hot(
                zero_indexed_groundtruth_classes, self.num_classes)
            
            return image_tensor, boxes_tensor, gt_classes_one_hot_tensor

        def set_shapes(image, boxes, classes):
            image.set_shape((None, None, 3))
            boxes.set_shape((None, 4))
            classes.set_shape((None, self.num_classes))
            
            return image, boxes, classes

        file_list = tfds.list_files(file_path_pattern)
        self.frames = file_list.map(lambda x: tf.py_function(prepare_frames, [x], [tf.float32, tf.float32, tf.float32]))
        self.frames = self.frames.map(set_shapes)
        self.frames = self.frames.shuffle(1000, reshuffle_each_iteration=True)

        print('Done prepping dataset.')