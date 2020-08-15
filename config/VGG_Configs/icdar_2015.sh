BATCH_SIZE=4
NUM_WORKERS=3
INPUT_SIZE=512
LEARNING_RATE=1e-4
NUM_STEPS=100000
POWER=0.9
SAVE_PRED_EVERY=5000
SNAPSHOT_DIR='./snapshots/'
DATASET='ICDAR2015'
CHECKPOINT_NO=55000
UPDATE_VISDOM_ITER=100
BACKBONE='VGG'
CHECKPOINT_PATH='snapshots/batch_size_4lr_0.0001n_steps_100000dataset_ICDAR2015backbone_VGG/ICDAR2015_3d_rotated_gaussian_without_attention_55000.pth'
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
