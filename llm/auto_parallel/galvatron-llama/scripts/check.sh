set -x

ProfileDataParserArgs="
    --time_profile_mode batch \
    --memory_profile_mode static \
    --num_layertype 1 \
    --hidden_size_list 4096 \
    --layernum_list 16 \
    --seqlen_list 1024 \
    --profile_gpu_num 4 \
    --time_profile_data_path ./configs/computation_profiling_bf16_llama_rank[0].json \
    --memory_profile_data_path ./configs/memory_profiling_bf16_llama.json \
    --overlap_coe_path ./configs/overlap_coefficient.json \
    --allreduce_coe_path ./configs/allreduce_bandwidth_1nodes_8gpus_per_node.json \
    --p2p_coe_path ./configs/p2p_bandwidth_1nodes_8gpus_per_node.json \
"

CostModelTrainArgs="
    --strategy pp1_tp4_dp1_stage0_recompute0 \
    --global_batch_size 64 \
    --mixed_precision_type bf16 \
    --accumulation_steps 16 \
"

python ./check.py ${ProfileDataParserArgs} ${CostModelTrainArgs}