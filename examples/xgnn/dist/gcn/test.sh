#脚本详细说明见examples/xgnn/dist/graphsage/README.md
#注意修改脚本中各个文件的路径

#ip_config建在当前目录下，内容是分布式节点的ip地址，如：
#10.244.68.63
#10.244.68.57

#数据分区：（在examples/xgnn/dist路径下）
#   python partition_graph.py --dataset ogbn-products --num_parts 2 --output /home/data

/root/anaconda3/envs/xgnn/bin/python /data/xgnn_distgpu/xgnn/utils/dgl_launch.py --workspace /data/xgnn_distgpu/examples/xgnn/dist/gcn \
--num_trainers 1 \
--num_samplers 0 \
--num_servers 1 \
--num_omp_threads 16 \
--part_config /data/data/xgnn/ogbn-products/random/2part_data/ogbn-products.json \
--ip_config ip_config.txt \
"/root/anaconda3/envs/xgnn/bin/python main.py --graph_name ogbn-products --part_config /data/data/xgnn/ogbn-products/random/2part_data/ogbn-products.json --ip_config ip_config.txt --num_gpus 2 --num_epochs 5 --eval_every 2 --num_hidden 16 --num_warmup 6"
