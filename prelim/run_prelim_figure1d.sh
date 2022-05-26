echo "====MLP after k layers propagation===="
for prop_k in 0 1 2 3 4
do
    echo "====prediction after k layer propagation====" $prop_k
    python main.py --dataset='german' --model='MLP' --top_k=0 --order='s2' --prop_k=$prop_k --use_feature='after_prop' --runs=10
done
