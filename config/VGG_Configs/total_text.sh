BATCH_SIZE=2
NUM_WORKERS=3
INPUT_SIZE=512
LEARNING_RATE=1e-4
NUM_STEPS=100000
POWER=0.9
SAVE_PRED_EVERY=1000
SNAPSHOT_DIR='./snapshots/'
DATASET='TOTALTEXT'
CHECKPOINT_NO=0
UPDATE_VISDOM_ITER=10
BACKBONE='VGG'
CHECKPOINT_PATH='snapshots/SynthText_3d_rotated_gaussian_without_attention_200000.pth'
ITERATION_TO_START_FROM=`expr $CHECKPOINT_NO + 1`
CUDA_DEVICE=0

nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9


CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_modular.py --batch-size=$BATCH_SIZE \
            --learning-rate=$LEARNING_RATE \
            --num-steps=$NUM_STEPS \
            --save-pred-every=$SAVE_PRED_EVERY \
            --snapshot-dir=$SNAPSHOT_DIR \
            --input-size=$INPUT_SIZE \
            --dataset=$DATASET \
            --checkpoint=$CHECKPOINT_PATH \
            --iteration-to-start-from=$ITERATION_TO_START_FROM \
            --update-visdom-iter=$UPDATE_VISDOM_ITER\
            --backbone=$BACKBONE\
             --visualization
