# MODIFY
export CUDA_VISIBLE_DEVICES='6'
# MODIFY
num_gpus=1
# MODIFY
master_port=32355
# MODIFY
cfg=CONFIG

# MODIFY
batch_size=16
# MODIFY
epochs=200
# MODIFY
dataset=LEVIR-CD
# MODIFY
data_path=/data2/wuxiaomeng/$dataset

output=exps/$dataset/$batch_size
# MODIFY
tag=hglrmamba
# MODIFY
pretrained_path=/data1/wuxiaomeng/code/changemamba/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth

cd ./changedetection

python -m torch.distributed.launch \
        --nproc_per_node=$num_gpus \
        --master_port $master_port \
        train.py \
        --cfg $cfg \
        --dataset $dataset \
        --batch-size $batch_size \
        --data_path $data_path \
        --output $output  \
        --tag $tag\
        --epochs $epochs \
        --enable_amp \
        --pretrained $pretrained_path \
