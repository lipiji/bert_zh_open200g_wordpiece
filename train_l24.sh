data_path=/apdcephfs/share_916081/pjli/bert_zh_300g_wordpiece_base/data
ckpt_path=/apdcephfs/share_916081/pjli/bert_zh_300g_wordpiece_big/ckpt/
mkdir -p $ckpt_path
echo $(($INDEX*8))
echo $CHIEF_IP 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u train.py --embed_dim 1024 \
                      --ff_embed_dim 4096 \
                      --num_heads 16 \
                      --layers 24 \
                      --dropout 0.1 \
                      --train_data $data_path/train.txt \
                      --vocab $data_path/vocab.txt \
                      --min_occur_cnt 1000 \
                      --batch_size 32 \
                      --warmup_steps 10000 \
                      --lr 1e-4 \
                      --max_len 128 \
                      --world_size 32 \
                      --gpus 8 \
                      --MASTER_ADDR $CHIEF_IP \
                      --MASTER_PORT 29556 \
                      --start_rank $(($INDEX*8)) \
                      --print_every 100 \
                      --save_every 20000 \
                      --save_dir $ckpt_path \
                      --backend nccl
