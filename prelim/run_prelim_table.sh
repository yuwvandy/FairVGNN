echo "====german===="

echo "====MLP_origin===="
python main.py --dataset='german' --model='MLP' --top_k=0

echo "====MLP_s1_top4===="
python main.py --dataset='german' --model='MLP' --order='s1' --top_k=4 --type='top'

echo "====MLP_s2_top4===="
python main.py --dataset='german' --model='MLP' --order='s2' --top_k=4 --type='top'

echo "====GCN_origin===="
python main.py --dataset='german' --model='GCN' --top_k=0 --prop_k=1

echo "====GCN_s1_top4===="
python main.py --dataset='german' --model='GCN' --order='s1' --top_k=4 --type='top' --prop_k=1

echo "====GCN_s2_top4===="
python main.py --dataset='german' --model='GCN' --order='s2' --top_k=4 --type='top' --prop_k=1


echo "====GIN_origin===="
python main.py --dataset='german' --model='GIN' --top_k=0 --prop_k=1

echo "====GIN_s1_top4===="
python main.py --dataset='german' --model='GIN' --order='s1' --top_k=4 --type='top' --prop_k=1

echo "====GIN_s2_top4===="
python main.py --dataset='german' --model='GIN' --order='s2' --top_k=4 --type='top' --prop_k=1


echo "====credit===="
echo "====MLP_origin===="
python main.py --dataset='credit' --model='MLP' --top_k=0

echo "====MLP_s1_top4===="
python main.py --dataset='credit' --model='MLP' --order='s1' --top_k=4 --type='top'

echo "====MLP_s2_top4===="
python main.py --dataset='credit' --model='MLP' --order='s2' --top_k=4 --type='top'

echo "====GCN_origin===="
python main.py --dataset='credit' --model='GCN' --top_k=0 --prop_k=1

echo "====GCN_s1_top4===="
python main.py --dataset='credit' --model='GCN' --order='s1' --top_k=4 --type='top' --prop_k=1

echo "====GCN_s2_top4===="
python main.py --dataset='credit' --model='GCN' --order='s2' --top_k=4 --type='top' --prop_k=1
