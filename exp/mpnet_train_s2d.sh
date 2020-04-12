cd ..
python3 mpnet_train.py --model_path ./models/ \
--no_env 10 --no_motion_paths 400 --grad_step 1 --learning_rate 0.01 \
--num_epochs 1 --device 3  --start_epoch 0 --data_path data/ --world_size 20 --env_type s2d \
--total_input_size 2804 --AE_input_size 2800 --mlp_input_size 32 --output_size 2
cd exp


#100x4000
