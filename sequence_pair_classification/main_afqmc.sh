task="afqmc"
ckpt_path=./ckpt/$task
mkdir -p $ckpt_path
python3 -u main.py \
    --bert_path ../ckpt/epoch1_batch_6919999 \
    --bert_vocab ../model/vocab.txt \
    --train_data ./data/$task/train.txt \
    --dev_data ./data/$task/dev.txt \
    --max_len 128 \
    --batch_size 10 \
    --lr 1e-5 \
    --dropout 0.2 \
    --number_class 2 \
    --number_epoch 20 \
    --gpu_id 0 \
    --print_every 500 \
    --fine_tune \
    --model_save_path  $ckpt_path
   
