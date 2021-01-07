#!/bin/bash
# Download model and checkpoint
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
tar -xf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
if [ -d "models/research/object_detection/test_data/checkpoint" ]; then rm -Rf models/research/object_detection/test_data/checkpoint; fi
mkdir models/research/object_detection/test_data/checkpoint
mv ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint models/research/object_detection/test_data/