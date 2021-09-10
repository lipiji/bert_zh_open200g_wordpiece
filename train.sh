data_path=/apdcephfs/share_916081/pjli/bert_zh_300g_wordpiece_base/data
ckpt_path=./ckpt/
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 train.py --embed_dim 768 \
                      --ff_embed_dim 3072 \
                      --num_heads 12 \
                      --layers 12 \
                      --dropout 0.1 \
                      --train_data $data_path/train.txt \
                      --vocab $data_path/vocab.txt \
                      --min_occur_cnt 1000 \
                      --batch_size 2 \
                      --warmup_steps 10000 \
                      --lr 1e-4 \
                      --max_len 128 \
                      --world_size 4 \
                      --gpus 4 \
                      --MASTER_ADDR localhost \
                      --MASTER_PORT 29556 \
                      --start_rank 0 \
                      --print_every 100 \
                      --save_every 20000 \
                      --save_dir $ckpt_path \
                      --backend nccl
