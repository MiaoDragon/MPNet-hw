cd ..
python3 cmpnet_train.py --model_path /media/arclabdl1/HD1/YLmiao/CMPnet_res/s2d/ \
--no_env 100 --no_motion_paths 4000 --grad_step 1 --learning_rate 0.01 \
--num_epochs 1 --memory_strength 0.5 --n_memories 10000 \
--n_tasks 1 --device 3 --freq_rehersal 100 --batch_rehersal 100 \
--start_epoch 0 --data_path /media/arclabdl1/HD1/YLmiao/mpnet/data/simple/ --world_size 20 --env_type s2d \
--memory_type res --total_input_size 2804 --AE_input_size 2800 --mlp_input_size 32 --output_size 2
cd exp


#100x4000
