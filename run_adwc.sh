echo '==========GCN-spmm-wc - 0.1=========='
python fairvgnn.py --dataset='german' --encoder='GCN' --clip_e=0.1 --d_epochs=5 --g_epochs=5 --c_epochs=5 --c_lr=0.01 --e_lr=0.001 --ratio=0.5 --epochs=400 --prop='spmm'

echo '==========GCN-spmm-wc - 2=========='
python fairvgnn.py --dataset='german' --encoder='GCN' --clip_e=2 --d_epochs=10 --g_epochs=10 --c_epochs=10 --c_lr=0.01 --e_lr=0.01 --ratio=0.5 --prop='spmm'

echo '==========GCN-spmm-wc - 4=========='
python fairvgnn.py --dataset='german' --encoder='GCN' --clip_e=4 --d_epochs=10 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.01 --ratio=0 --prop='spmm'

echo '==========GCN-spmm-wc - 6=========='
python fairvgnn.py --dataset='german' --encoder='GCN' --clip_e=6 --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.01 --ratio=1 --prop='spmm'

echo '==========GCN-spmm-wc - 8=========='
python fairvgnn.py --dataset='german' --encoder='GCN' --clip_e=8 --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.01 --ratio=0.5 --prop='spmm'
