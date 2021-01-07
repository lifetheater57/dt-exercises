#!/bin/bash
# Install requirements
cd ./models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install setuptools>=39.2.0
python -m pip install --no-cache-dir .