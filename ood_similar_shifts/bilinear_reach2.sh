GPU_DEVICE=1
for layer_depth in 1
do
    for layer_size in 32 512 1024
    do
        echo 'layer_size '$layer_size' layer_depth '$layer_depth
        # GPU_DEVICE=0
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/reach_bilinear.py \
        --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
        --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
        --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
        --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size #&
        # GPU_DEVICE=1
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/reach_bilinear.py --fourier \
        --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
        --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
        --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
        --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size #&
    done
done

