#!/bin/bash

# $1 : Input directory where model.ckpt.* having SSD detection model resides
# $2 : Output path to save the updated model with feature accessors
# $3 : Pipeline config path, e.g., object_detection/samples/configs/ssd_mobilenet_v1_coco.config

cd research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
protoc object_detection/protos/*.proto --python_out=.
python object_detection/export_inference_graph_ssd.py \
  --input_type image_tensor \
  --pipeline_config_path $3 \
  --trained_checkpoint_prefix $1/model.ckpt \
  --output_directory $2
cd ..

