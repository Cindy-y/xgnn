script_dir=$(cd $(dirname $0);pwd)

python_path=$(dirname $script_dir)

export PYTHONPATH=$python_path

cd $python_path

# xgnn data processing
python examples/xgnn/partition_graph.py --dataset ogbn-products --num_parts 2 --part_method random
# python examples/xgnn/partition_graph.py --dataset ogbn-products --num_parts 3 --part_method random

# python examples/xgnn/partition_graph.py --dataset ogbn-products --num_parts 4 --part_method random

# distdal data processing