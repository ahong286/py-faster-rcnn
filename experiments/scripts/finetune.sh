#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=0
NET=VGG16
DATASET=dleb17

ITERS=50000
NUM_CLASSES=4

TRAIN_IMDB="${DATASET}_train"
TEST_IMDB="${DATASET}_test"

FRCNN=$(pwd)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Create network directory for the input dataset.
# Copy a well-defined network and make modification based on it
NET_DIR=${FRCNN}/models/${DATASET}/${NET}
mkdir -p ${NET_DIR}
cp ${FRCNN}/models/pascal_voc/${NET}/faster_rcnn_end2end/* ${NET_DIR}/
cd ${NET_DIR}

# Modify solver.prototxt
sed -i "s|models/pascal_voc/${NET}/faster_rcnn_end2end/train.prototxt|${NET_DIR}/train.prototxt|g"  solver.prototxt

# Modify train and test prototxt files.
## Modify number of classes
grep -rl "num_output: 21" ./ | xargs sed -i "s|num_output: 21|num_output: ${NUM_CLASSES}|g"

## Modify number of classes
I_STRING="param_str: "\"\'"num_classes"\'": 21"\"
O_STRING="param_str: "\"\'"num_classes"\'": "${NUM_CLASSES}\"
grep -rl "'num_classes': 21" | xargs sed -i  "s/${I_STRING}/${O_STRING}/g"

## Modify number of bounding boxes, ie n_classes * 4
grep -rl "num_output: 84" | xargs sed -i "s|num_output: 84|num_output: $(( ${NUM_CLASSES} * 4 ))|g"

# Modify train.prototxt to change class_score and bbox_score to finetune the network correctly.
## Modify cls_score
I_STRING=\""cls_score"\"
O_STRING=\""cls_score_f"\"
sed -i "s/${I_STRING}/${O_STRING}/g" train.prototxt

## Modify bbox_pred
I_STRING=\""bbox_pred"\"
O_STRING=\""bbox_pred_f"\"
sed -i "s/${I_STRING}/${O_STRING}/g" train.prototxt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# 2. Finetuning I
NET_INIT="data/imagenet_models/${NET}.v2.caffemodel"
TMP_LOG=$(mktemp)
cd ${FRCNN}
./tools/train_net.py \
    --gpu ${GPU_ID} \
    --solver models/${DATASET}/${NET}/solver.prototxt \
    --weights ${NET_INIT} \
    --imdb ${TRAIN_IMDB} \
    --iter 0 \
    --cfg experiments/cfgs/config.yml | tee -a "${TMP_LOG}"

set +x
NET_INIT=`grep -B 1 "done solving" ${TMP_LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x
rm ${TMP_LOG}

# Modify cls_score back
cd ${NET_DIR}
I_STRING=\""cls_score_f"\"
O_STRING=\""cls_score"\"
sed -i "s/${I_STRING}/${O_STRING}/g" train.prototxt

## Modify bbox_pred back
I_STRING=\""bbox_pred_f"\"
O_STRING=\""bbox_pred"\"
sed -i "s/${I_STRING}/${O_STRING}/g" train.prototxt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# 3. Finetuning II
TRN_LOG="experiments/logs/faster_rcnn_end2end_${TRAIN_IMDB}_${NET}.txt"
cd ${FRCNN}
time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${DATASET}/${NET}/solver.prototxt \
  --weights ${NET_INIT} \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/config.yml| tee -a "${TRN_LOG}"

set +x
NET_FINAL=`grep -B 1 "done solving" ${TRN_LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

TST_LOG="experiments/logs/faster_rcnn_end2end_${TEST_IMDB}_${NET}.txt"
time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${DATASET}/${NET}/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/config.yml| tee -a "${TST_LOG}"
