cd ../src/models
python train_multi_tmp.py --training_file_path "./train_data/*"  --log_path log_path --embedding_file_path data --batch_size 128 --history_aggregate_mode 0 --use_group_attention True --user_activation tanh 
cd -
