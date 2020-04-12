cd ..
# End-2-End learning (randomly shuffle path)
python3 mpnet_train.py --model_path ../MPnet_res/r2d/ \
--no_env 100 --no_motion_paths 4000 --grad_step 1 --learning_rate 0.01 \
--num_epochs 100 --memory_strength 0.5 --n_memories 10000 \
--n_tasks 1 --device 0 --freq_rehersal 100 --batch_rehersal 100 \
--start_epoch 0 --data_path /media/arclabdl1/HD1/Ahmed/rigid-body/dataset/ --world_size 20 --env_type r2d \
--memory_type res --total_input_size 2806 --AE_input_size 2800 --mlp_input_size 34 --output_size 3
cd exp
