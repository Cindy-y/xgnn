/home/nx/anaconda3/envs/distdgl/bin/python /home/nx/ningxin/DistGNN/xgnn/xgnn/utils/launch.py   \
		--workspace /home/nx/ningxin/DistGNN/xgnn  \
		--num_trainers 1   \
		--num_samplers 0   \
		--num_servers 1   \
		--part_config /home/data/xgnn/ogbn-products/random/2part_data/ogbn-products.json   \
		--ip_config examples/ip_config.txt  \
		"docker exec --env PYTHONPATH=/home/xgnn --env NCCL_IB_DISABLE=1 --env NCCL_DEBUG=INFO -w /home/xgnn xgnn /opt/conda/bin/python examples/xgnn/dist/graphsage/main_without_pipeline.py \
        --graph_name ogbn-products --ip_config examples/ip_config.txt \
		--part_config /home/data/xgnn/ogbn-products/random/2part_data/ogbn-products.json --num_epochs 5 --eval_every 2 --num_hidden 16"
# --env NCCL_DEBUG_SUBSYS=ALL