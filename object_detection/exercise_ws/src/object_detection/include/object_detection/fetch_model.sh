#!/bin/bash
# This scripts fetches the base model and prepares it.
if [ ! -e "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz" ]
    then 
        wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
fi
tar -xf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
if [ -d "models/test_data/checkpoint" ]; then rm -Rf models/test_data/checkpoint; fi
mkdir models/test_data/checkpoint
mv ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint models/test_data/