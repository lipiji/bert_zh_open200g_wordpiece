data_path=./data
ckpt_path=./ckpt/
mkdir -p $ckpt_path

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u train.py --embed_dim 768 \
                      --ff_embed_dim 3072 \
                      --num_heads 12 \
                      --layers 12 \
                      --dropout 0.1 \
                      --train_data $data_path/train.txt \
                      --vocab $data_path/vocab.txt \
                      --min_occur_cnt 1200 \
                      --batch_size 64 \
                      --warmup_steps 10000 \
                      --lr 1e-4 \
                      --accumulation_steps 4 \
                      --max_len 128 \
                      --world_size 8 \
                      --gpus 8 \
                      --MASTER_ADDR localhost \
                      --MASTER_PORT 29556 \
                      --start_rank 0 \
                      --print_every 100 \
                      --save_every 10000 \
                      --save_dir $ckpt_path \
                      --backend nccl
                      #--start_from ./ckpt/epoch2_batch_2359999
