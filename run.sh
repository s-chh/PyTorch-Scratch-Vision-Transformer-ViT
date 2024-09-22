python -u main.py --data_path /home/local/ASUAD/schhabr6/DB/ --dataset mnist --epochs 100 &> ./results/mnist.txt
python -u main.py --data_path /home/local/ASUAD/schhabr6/DB/ --dataset fmnist --embed_dim 64 &> ./results/fmnist.txt
python -u main.py --data_path /home/local/ASUAD/schhabr6/DB/ --dataset svhn --embed_dim 128 --n_channels 3 --image_size 32 &> ./results/svhn.txt
python -u main.py --data_path /home/local/ASUAD/schhabr6/DB/ --dataset cifar10 --embed_dim 128 --n_channels 3 --image_size 32 &> ./results/cifar10.txt
python -u main.py --data_path /home/local/ASUAD/schhabr6/DB/ --dataset cifar100 --embed_dim 128 --n_channels 3 --image_size 32 --n_classes 100 &> ./results/cifar100.txt
