# train_multilabel_final.py --model_name_or_path bert-base-uncased --train_file ../data/train.json --validation_file ../data/dev.json --test_file ../data/test.json --output_dir ../output --do_train --do_eval --do_predict --num_train_epochs 3 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate 2e-5 --logging_steps 100 --save_steps 100 --seed 42 --overwrite_output_dir --max_seq_length 128 --task_name multilabel --label_list ../data/labels.txt --weight_decay 0.01 --warmup_steps 500 --logging_dir ../logs --logging_first_step --save_total_limit 2 --fp16
# train_multilabel_final.py -c ../output/checkpoint-1000 --model_name_or_path bert-base-uncased --train_file ../data/train.json --validation_file ../data/dev.json --test_file ../data/test.json --output_dir ../output --do_train --do_eval --do_predict --num_train_epochs 3 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate 2e-5 --logging_steps 100 --save_steps 100 --seed 42 --overwrite_output_dir --max_seq_length 128 --task_name multilabel --label_list ../data/labels.txt --weight_decay 0.01 --warmup_steps 500 --logging_dir ../logs --logging_first_step --save_total_limit 2 --fp16


# train.py -c config1.yaml
# train.py -c config2.yaml
# train.py -c config3.yaml

# ./script.sh 실행.

python train_multilabel_final.py --config=config1.json
python train_multilabel_final.py --config=config2.json
python train_multilabel_final.py --config=config3.json
python train_multilabel_final.py --config=config4.json
python train_multilabel_final.py --config=config5.json
python train_multilabel_final.py --config=config6.json
python train_multilabel_final.py --config=config7.json
