# Fill the parameters below
################################################################################
VOC_ROOT="data/VOC2007/train/VOCdevkit/VOC2007"
VOC_ROOT_VALID="data/VOC2007/test/VOCdevkit/VOC2007"

SAVE_DIR_MODEL="models/keras-VOC"
################################################################################

# Also, you can modify the training hyperparameters
python -m efficientdet.train \
    --bidirectional \
    --no-freeze-backbone \
    --print-freq 100 \
    --validate-freq 10 \
    --epochs 100 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --w-scheduler \
    --save-dir $SAVE_DIR_MODEL \
    VOC \
    --root-train $VOC_ROOT \
    --root-valid $VOC_ROOT_VALID