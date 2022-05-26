echo '***********************GERMAN***********************'
echo '==========GCN-spmm=========='
python fairvgnn.py --dataset='german' --encoder='GCN' --clip_e=0.1 --d_epochs=5 --g_epochs=5 --c_epochs=5 --c_lr=0.01 --e_lr=0.001 --ratio=0.5 --epochs=400 --prop='spmm'

echo '==========GCN-spmm wo fm=========='
python fairvgnn.py --dataset='german' --encoder='GCN' --clip_e=1 --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.01 --f_mask='no' --prop='spmm'

echo '==========GCN-spmm wo wc=========='
python fairvgnn.py --dataset='german' --encoder='GCN' --d_epochs=5 --g_epochs=10 --c_epochs=10 --c_lr=0.01 --e_lr=0.01 --ratio=0 --weight_clip='no' --prop='spmm'

echo '==========GCN-spmm wo fm&wc=========='
python fairvgnn.py --dataset='german' --encoder='GCN' --d_epochs=5 --g_epochs=10 --c_epochs=10 --c_lr=0.01 --e_lr=0.01 --f_mask='no' --weight_clip='no' --epochs=400 --prop='spmm'





echo '==========GCN-scatter=========='
python fairvgnn.py --dataset='german' --encoder='GCN' --clip_e=0.1 --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.001 --ratio=0

echo '==========GIN-scatter wo fm=========='
python fairvgnn.py --dataset='german' --encoder='GCN' --clip_e=1 --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.01 --f_mask='no'

echo '==========GIN-scatter wo wc=========='
python fairvgnn.py --dataset='german' --encoder='GCN' --d_epochs=10 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.01 --ratio=0.5 --weight_clip='no'

echo '==========GIN-scatter wo fm&wc=========='
python fairvgnn.py --dataset='german' --encoder='GCN' --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.01 --f_mask='no' --weight_clip='no'





echo '==========GIN=========='
python fairvgnn.py --dataset='german' --encoder='GIN' --clip_e=1 --d_epochs=10 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.01 --ratio=1

echo '==========GIN wo fm=========='
python fairvgnn.py --dataset='german' --encoder='GIN' --clip_e=1 --d_epochs=10 --g_epochs=10 --c_epochs=5 --c_lr=0.001 --e_lr=0.001 --f_mask='no'

echo '==========GIN wo wc=========='
python fairvgnn.py --dataset='german' --encoder='GIN' --d_epochs=10 --g_epochs=10 --c_epochs=10 --c_lr=0.01 --e_lr=0.001 --ratio=0 --weight_clip='no'

echo '==========GIN wo fm&wc=========='
python fairvgnn.py --dataset='german' --encoder='GIN' --d_epochs=5 --g_epochs=10 --c_epochs=5 --c_lr=0.01 --e_lr=0.01 --f_mask='no' --weight_clip='no'





echo '==========SAGE=========='
python fairvgnn.py --dataset='german' --encoder='SAGE' --clip_e=0.1 --d_epochs=5 --g_epochs=10 --c_epochs=10 --c_lr=0.001 --e_lr=0.001 --ratio=0.5

echo '==========SAGE wo fm=========='
python fairvgnn.py --dataset='german' --encoder='SAGE' --clip_e=0.1 --d_epochs=5 --g_epochs=10 --c_epochs=10 --c_lr=0.001 --e_lr=0.001 --f_mask='no'

echo '==========SAGE wo wc=========='
python fairvgnn.py --dataset='german' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=10 --c_lr=0.001 --e_lr=0.01 --ratio=0.5 --weight_clip='no'

echo '==========SAGE w/o fm&wc=========='
python fairvgnn.py --dataset='german' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=10 --c_lr=0.001 --e_lr=0.001 --f_mask='no' --weight_clip='no'
