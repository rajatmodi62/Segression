#NUM_WORKERS=3
#INPUT_SIZE=512
DATASET='TOTALTEXT'
#BACKBONE='VGG'
# TEST_DIR="data/debug/total-text-original/Images/Test"
SEGRESSION_CHECKPOINT_PATH='snapshots/TOTALTEXT_3d_rotated_gaussian_without_attention_100000.pth'
LSTM_CHECKPOINT_PATH='snapshots/LSTM_batch_size_8lr_0.0001n_steps_100000dataset_TOTALTEXTbackbone_VGG/TOTALTEXT_LSTM_checkpoint6000.pth'
CUDA_DEVICE=0
SEGMENTATION_THRESHOLD=0.8
GAUSSIAN_THRESHOLD=0.65
LSTM_THRESHOLD=0.70
N_CLASSES=3
# MODE='with_lstm'
nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m testing_obsolete.testing_modular_lstm \
            --test-dir=$TEST_DIR \
            --snapshot-dir=$SEGRESSION_CHECKPOINT_PATH \
            --lstm-snapshot-dir=$LSTM_CHECKPOINT_PATH \
            --segmentation_threshold=$SEGMENTATION_THRESHOLD \
            --gaussian_threshold=$GAUSSIAN_THRESHOLD \
            --lstm_threshold=$LSTM_THRESHOLD \
            --n-classes=$N_CLASSES \
        #    --mode=$MODE
#--input-size=$INPUT_SIZE \
#--dataset=$DATASET \
#--backbone=$BACKBONE\