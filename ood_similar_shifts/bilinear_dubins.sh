# CUDA_VISIBLE_DEVICES=3 python ood_similar_shifts/dubins_bilinear.py
# CUDA_VISIBLE_DEVICES=3 python ood_similar_shifts/dubins_bilinear.py --fourier

GPU_DEVICE=6
# #debugging
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/dubins_bilinear.py \
# --bc-hidden-depth 0 --deep-sets-hidden-depth 0 --NN-hidden-depth 0 --bilinear-hidden-depth 0 \
# --bc-n-epochs 10 --deep-sets-n-epochs 10 --NN-n-epochs 10 --bilinear-n-epochs 10 \
# --expert-n-traj 100 --eval-n-traj 20 

# #linear
echo 'linear'
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/dubins_bilinear.py \
--bc-hidden-depth 0 --deep-sets-hidden-depth 0 --NN-hidden-depth 0 --bilinear-hidden-depth 0 &

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/dubins_bilinear.py --fourier \
--bc-hidden-depth 0 --deep-sets-hidden-depth 0 --NN-hidden-depth 0 --bilinear-hidden-depth 0 &

# GPU_DEVICE=3
# for layer_depth in 1
# do
#     for layer_size in 32 512 1024
#     do
#         echo 'layer_size '$layer_size' layer_depth '$layer_depth
        
#         CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/dubins_bilinear.py \
#         --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
#         --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
#         --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
#         --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &

#         CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/dubins_bilinear.py --fourier \
#         --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
#         --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
#         --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
#         --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &
#     done
# done


# GPU_DEVICE=4
# for layer_depth in 2
# do
#     for layer_size in 32 512 1024
#     do
#         echo 'layer_size '$layer_size' layer_depth '$layer_depth
        
#         CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/dubins_bilinear.py \
#         --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
#         --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
#         --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
#         --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &

#         CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/dubins_bilinear.py --fourier \
#         --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
#         --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
#         --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
#         --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &
#     done
# done