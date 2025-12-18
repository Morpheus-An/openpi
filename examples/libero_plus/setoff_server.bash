uv run scripts/serve_policy.py \
    --env LIBERO \
    policy:checkpoint \
    --policy.config pi0_libero_low_mem_finetune \
    --policy.dir /mars_data_2/ant/openpi/checkpoints/pi0_libero_low_mem_finetune/1212pi0_libero_spatial_lora_jax/29999

# run server_policy with torch model
# uv run scripts/serve_policy.py \
#     --env LIBERO \
#     policy:checkpoint \
#     --policy.config pi0_libero_lora_pytorch \
#     --policy.dir /mars_data_2/ant/openpi/checkpoints/pi0_libero_lora_pytorch/both_lora_toch_pi0_libero/12000
