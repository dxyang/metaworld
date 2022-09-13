GPU_DEVICE=0
for layer_depth in 1
do
    for layer_size in 32
    do
        echo 'layer_size '$layer_size' layer_depth '$layer_depth
        
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/franka_reach_bilinear.py \
        --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
        --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
        --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
        --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &

        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/franka_reach_bilinear.py --fourier \
        --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
        --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
        --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
        --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &
    done
done

GPU_DEVICE=1
for layer_depth in 1
do
    for layer_size in 512
    do
        echo 'layer_size '$layer_size' layer_depth '$layer_depth
        
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/franka_reach_bilinear.py \
        --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
        --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
        --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
        --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &

        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/franka_reach_bilinear.py --fourier \
        --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
        --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
        --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
        --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &
    done
done

GPU_DEVICE=2
for layer_depth in 1
do
    for layer_size in 1024
    do
        echo 'layer_size '$layer_size' layer_depth '$layer_depth

        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/franka_reach_bilinear.py \
        --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
        --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
        --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
        --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &

        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/franka_reach_bilinear.py --fourier \
        --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
        --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
        --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
        --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &
    done
done

#####
GPU_DEVICE=3
for layer_depth in 2
do
    for layer_size in 32
    do
        echo 'layer_size '$layer_size' layer_depth '$layer_depth
        
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/franka_reach_bilinear.py \
        --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
        --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
        --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
        --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &
        
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/franka_reach_bilinear.py --fourier \
        --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
        --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
        --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
        --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &
    done
done

GPU_DEVICE=4
for layer_depth in 2
do
    for layer_size in 512
    do
        echo 'layer_size '$layer_size' layer_depth '$layer_depth
        
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/franka_reach_bilinear.py \
        --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
        --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
        --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
        --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &
        
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/franka_reach_bilinear.py --fourier \
        --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
        --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
        --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
        --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &
    done
done

GPU_DEVICE=5
for layer_depth in 2
do
    for layer_size in 1024
    do
        echo 'layer_size '$layer_size' layer_depth '$layer_depth
        
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/franka_reach_bilinear.py \
        --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
        --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
        --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
        --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &
        
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/franka_reach_bilinear.py --fourier \
        --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
        --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
        --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
        --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &
    done
done