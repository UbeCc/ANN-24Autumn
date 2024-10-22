python main.py \
    --runid Tfmr_pretrain_post_norm \

python main.py \
    --runid Tfmr_pretrain_rms_norm \

python main.py \
    --runid GQA_pretrain \

# python main.py \
#     --name from_ckpt \
#     --pretrain_dir ./ckpt/model_3_layers/

# STRATEGY=random
# python main.py \

# STRATEGY=top-p
# python main.py \
#     --name
#     --decode_strategy
#     --top_p 
#     --temperature

# for decode_strategy in random top-p; do
#     for temperature in 0.7 0.9; do
#         python main.py \
#             --test Tfmr_finetune \
#             --runid finetune_${decode_strategy}_top_p_0.9_temperature_${temperature} \
#             --decode_strategy ${decode_strategy} \
#             --top_p 0.9 \
#             --temperature ${temperature} \
#             --train_dir train_test
#         echo Running pretrain_${decode_strategy}_top_p_0.9_temperature_${temperature} ...
#     done
# done

# for decode_strategy in random top-p; do
#     for temperature in 0.7 0.9; do
#         python main.py \
#             --test Tfmr_pretrain \
#             --runid pretrain_${decode_strategy}_top_p_0.9_temperature_${temperature} \
#             --decode_strategy ${decode_strategy} \
#             --top_p 0.9 \
#             --temperature ${temperature} \
#             --pretrain_dir ./ckpt/model_3_layers/ \
#             --train_dir train_test
#         echo Running pretrain_${decode_strategy}_top_p_0.9_temperature_${temperature} ...
#     done
# done