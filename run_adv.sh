echo '***********************GERMAN***********************'
echo '==========GCN-spmm wo wc=========='
python fairvgnn.py --dataset='german' --encoder='GCN' --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.01 --ratio=0 --weight_clip='no' --prop='spmm'

echo '==========GCN-spmm wo wc&d=========='
# we set alpha = 0.3 here to ensure the utility of the differerent baselines are of the same level so that comparing their bias is meaningful
python fairvgnn.py --dataset='german' --encoder='GCN' --d_epochs=0 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.01 --ratio=1 --weight_clip='no' --prop='spmm' --alpha=0.3

echo '==========GCN-spmm wo wc&g=========='
python fairvgnn.py --dataset='german' --encoder='GCN' --d_epochs=5 --g_epochs=0 --c_epochs=10 --c_lr=0.01 --e_lr=0.01 --ratio=0 --weight_clip='no' --prop='spmm'

echo '***********************CREDIT***********************'
echo '==========GCN-spmm wo wc=========='
python fairvgnn_credit.py --dataset='credit' --encoder='GCN' --clip_e=1 --d_epochs=10 --g_epochs=10 --c_epochs=5 --weight_clip='no' --c_lr=0.01 --e_lr=0.01 --epochs=200 --prop='spmm'

echo '==========GCN-spmm wo wc&d=========='
python fairvgnn_credit.py --dataset='credit' --encoder='GCN' --clip_e=1 --d_epochs=0 --g_epochs=10 --c_epochs=5 --weight_clip='no' --c_lr=0.01 --e_lr=0.01 --epochs=200 --prop='spmm'

echo '==========GCN-spmm wo wc&g==========' $alpha
python fairvgnn_credit.py --dataset='credit' --encoder='GCN' --clip_e=1 --d_epochs=10 --g_epochs=0 --c_epochs=5 --weight_clip='no' --c_lr=0.01 --e_lr=0.01 --epochs=100 --prop='spmm'
