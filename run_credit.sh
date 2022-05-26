echo '***********************credit***********************'
echo '==========GCN==========' pass
python fairvgnn_credit.py --dataset='credit' --encoder='GCN' --clip_e=1 --d_epochs=5 --g_epochs=10 --c_epochs=5 --c_lr=0.01 --e_lr=0.01 --ratio=0 --epochs=200

echo '==========GCN w/o fm==========' pass
python fairvgnn_credit.py --dataset='credit' --encoder='GCN' --clip_e=1 --d_epochs=5 --g_epochs=10 --c_epochs=5 --c_lr=0.01 --e_lr=0.01 --f_mask='no' --epochs=200

echo '==========GCN w/o wc==========' pass
python fairvgnn_credit.py --dataset='credit' --encoder='GCN' --clip_e=1 --d_epochs=10 --g_epochs=10 --c_epochs=5 --weight_clip='no' --c_lr=0.01 --e_lr=0.01 --epochs=200

echo '==========GCN w/o fm&wc==========' pass
python fairvgnn_credit.py --dataset='credit' --encoder='GCN' --d_epochs=5 --g_epochs=5 --c_epochs=5 --c_lr=0.01 --e_lr=0.01 --f_mask='no' --weight_clip='no' --epochs=200





echo '==========GIN==========' pass
python fairvgnn_credit.py --dataset='credit' --encoder='GIN' --clip_c=0.1 --clip_e=0.01 --d_epochs=10 --g_epochs=10 --c_epochs=5 --epochs=100 --c_lr=0.01 --e_lr=0.01

echo '==========GIN w/o fm==========' pass
python fairvgnn_credit.py --dataset='credit' --encoder='GIN' --clip_c=0.1 --clip_e=0.01 --d_epochs=5 --g_epochs=10 --c_epochs=10 --f_mask='no' --epochs=200 --c_lr=0.01 --e_lr=0.01

echo '==========GIN w/o wc=========='
python fairvgnn_credit.py --dataset='credit' --encoder='GIN' --clip_c=10 --clip_e=10 --d_epochs=5 --g_epochs=5 --c_epochs=20 --f_mask='no' --epochs=500 --c_lr=0.01 --e_lr=0.01

echo '==========GIN w/o fm&wc==========' pass
python fairvgnn_credit.py --dataset='credit' --encoder='GIN' --clip_c=10 --clip_e=10 --d_epochs=5 --g_epochs=5 --c_epochs=10 --f_mask='no' --weight_clip='no' --epochs=200 --c_lr=0.01 --e_lr=0.01





echo '==========SAGE==========' pass
python fairvgnn_credit.py --dataset='credit' --encoder='SAGE' --clip_e=0.1 --d_epochs=5 --g_epochs=5 --c_epochs=5 --c_lr=0.01 --e_lr=0.001 --ratio=0

echo '==========SAGE wo fm==========' pass
python fairvgnn_credit.py --dataset='credit' --encoder='SAGE' --clip_e=0.1 --clip_c=0.1 --d_epochs=5 --g_epochs=5 --c_epochs=5 --c_lr=0.01 --e_lr=0.01 --f_mask='no'

echo '==========SAGE w/o wc==========' pass
python fairvgnn_credit.py --dataset='credit' --encoder='SAGE' --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.001 --ratio=0 --weight_clip='no'

echo '==========SAGE w/o fm&wc==========' pass
python fairvgnn_credit.py --dataset='credit' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=5 --c_lr=0.01 --e_lr=0.01 --f_mask='no' --weight_clip='no'
