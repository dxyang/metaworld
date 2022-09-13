CUDA_VISIBLE_DEVICES=3 python ood_similar_shifts/wheeled_bilinear.py
CUDA_VISIBLE_DEVICES=3 python ood_similar_shifts/wheeled_bilinear.py --fourier
CUDA_VISIBLE_DEVICES=3 python ood_similar_shifts/wheeled_bilinear.py --bc-hidden-depth 1 --NN-hidden-depth 1
CUDA_VISIBLE_DEVICES=3 python ood_similar_shifts/wheeled_bilinear.py --bc-hidden-depth 2 --NN-hidden-depth 2

# CUDA_VISIBLE_DEVICES=3 python ood_similar_shifts/wheeled_bilinear.py --expert-n-traj 10 --bc-n-epochs 10 --bc-n-test-traj 10 --NN-n-epochs 10 --bilinear-hidden-depth 1
