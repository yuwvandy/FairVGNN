echo '***********************Bail***********************'
echo '==========GCN-scatter=========='
python fairvgnn.py --dataset='bail' --encoder='GCN' --clip_e=1 --d_epochs=5 --g_epochs=10 --c_epochs=10 --c_lr=0.01 --e_lr=0.001 --ratio=1 --epochs=300

echo '==========GCN-scatter wo fm==========' pass
python fairvgnn.py --dataset='bail' --encoder='GCN' --clip_e=1 --d_epochs=5 --g_epochs=5 --c_epochs=5 --c_lr=0.01 --e_lr=0.01 --f_mask='no'

echo '==========GCN-scatter wo wc=========='
python fairvgnn.py --dataset='bail' --encoder='GCN' --d_epochs=10 --g_epochs=10 --c_epochs=10 --c_lr=0.001 --e_lr=0.01 --ratio=1 --weight_clip='no'

echo '==========GCN-scatter wo fm&wc=========='
python fairvgnn.py --dataset='bail' --encoder='GCN' --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.001 --e_lr=0.001 --f_mask='no' --weight_clip='no'



echo '==========GCN-spmm=========='
python fairvgnn.py --dataset='bail' --encoder='GCN' --clip_e=1 --d_epochs=5 --g_epochs=10 --c_epochs=10 --c_lr=0.01 --e_lr=0.001 --ratio=1 --epochs=300 --prop='spmm'

echo '==========GCN-spmm wo fm=========='
python fairvgnn.py --dataset='bail' --encoder='GCN' --clip_e=1 --d_epochs=5 --g_epochs=5 --c_epochs=5 --c_lr=0.01 --e_lr=0.01 --f_mask='no' --prop='spmm'

echo '==========GCN-spmm wo wc=========='
python fairvgnn.py --dataset='bail' --encoder='GCN' --d_epochs=10 --g_epochs=10 --c_epochs=10 --c_lr=0.001 --e_lr=0.01 --ratio=1 --weight_clip='no' --prop='spmm'

echo '==========GCN-spmm wo fm&wc=========='
python fairvgnn.py --dataset='bail' --encoder='GCN' --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.001 --e_lr=0.001 --f_mask='no' --weight_clip='no' --prop='spmm'




echo '==========SAGE=========='
python fairvgnn.py --dataset='bail' --encoder='SAGE' --clip_e=0.1 --d_epochs=5 --g_epochs=5 --c_epochs=5 --c_lr=0.01 --e_lr=0.001 --ratio=1

echo '==========SAGE w/o wc=========='
python fairvgnn.py --dataset='bail' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=5 --c_lr=0.01 --e_lr=0.001 --ratio=0.5 --weight_clip='no'

echo '==========SAGE wo fm=========='
python fairvgnn.py --dataset='bail' --encoder='SAGE' --clip_e=1 --d_epochs=5 --g_epochs=10 --c_epochs=5 --c_lr=0.001 --e_lr=0.001 --f_mask='no'

echo '==========SAGE w/o fm&wc=========='
python fairvgnn.py --dataset='bail' --encoder='SAGE' --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.001 --e_lr=0.001 --f_mask='no' --weight_clip='no'
