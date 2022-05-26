echo "====german===="
echo "====MLP_origin===="
python main.py --dataset='german' --model='MLP' --top_k=0

echo "====MLP_s1===="
for top_k in 1 2 3 4
do
    python main.py --dataset='german' --model='MLP' --order='s1' --top_k=$top_k --type='single'
done

echo "====GCN_origin===="
python main.py --dataset='german' --model='GCN' --top_k=0

echo "====GCN_s1===="
for top_k in 1 2 3 4
do
    python main.py --dataset='german' --model='GCN' --order='s1' --top_k=$top_k --type='single'
done


