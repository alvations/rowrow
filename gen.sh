set -e

PATH_TO_MODEL=/home/ltan/rowrow/model

paddle train \
    --job=test \
    --config='gen.conf' \
    --save_dir=${PATH_TO_MODEL} \
    --use_gpu=true \
    --num_passes=1 \
    --test_pass=0 \
    --trainer_count=1 \
    2>&1 | tee 'gen.log'
