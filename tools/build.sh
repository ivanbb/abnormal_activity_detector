#!/bin/bash
cp ../app/config_base ../app/config -R
docker build . -t deepstream-abnormal-detections
