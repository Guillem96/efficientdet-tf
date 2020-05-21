# Fill the parameters below
################################################################################
VOC_ROOT=""
VOC_ROOT_VALID=""

SAVE_DIR_MODEL=""
################################################################################

# Also, you can modify the training hyperparameters
python -m efficientdet.train \
    --bidirectional \
    --no-freeze-backbone \
    --print-freq 100 \
    --validate-freq 10 \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 1e-3 \
    --w-scheduler \
    --save-dir $SAVE_DIR_MODEL \
    VOC \
    --root-train $VOC_ROOT \
    --root-valid $VOC_ROOT_VALID