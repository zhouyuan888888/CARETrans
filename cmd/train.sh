DATA_PATH=/path/to/your/dataset
CODE_PATH=/path/to/CARETrans
MODEL=CARETrans_S0
NUM_GPU=12
ALL_BATCH_SIZE=NUM_GPU*128
GRAD_ACCUM_STEPS=1
DROP_PATH=0.1
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS
MIN_FREE_MEMORY=10000
check_gpu_memory() {
    while IFS= read -r line; do
        free_memory=$(echo $line | awk '{ print $1 }')
        if [ "$free_memory" -lt "$MIN_FREE_MEMORY" ]; then
            return 1
        fi
    done <<< $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    return 0
}
loop=true
print=true
while ${loop}; do
    if check_gpu_memory; then
        echo "The gpu memory is higher than (${MIN_FREE_MEMORY}MiB)，so beginning!"
          cd ${CODE_PATH} && python3 -m torch.distributed.launch \
          --nproc_per_node=${NUM_GPU} \
          --master_addr="127.0.0.1" \
          --master_port=29601 \
          --use_env train.py \
          ${DATA_PATH} \
          --model ${MODEL} \
          --opt adamw \
          --lr 2.5e-3 \
          --warmup-epochs 20 \
          -b ${BATCH_SIZE} --grad-accum-steps ${GRAD_ACCUM_STEPS} \
          --drop-path ${DROP_PATH} \
          --output ./output
        break
    else
      if ${print}; then
            echo "There is at least, one gpu whose memory is lower than (${MIN_FREE_MEMORY}MiB), so waiting..."
            print=false
        fi
        sleep 10
    fi
done