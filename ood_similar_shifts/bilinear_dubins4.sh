GPU_DEVICE=0
for layer_depth in 2
do
    for layer_size in 512 1024
    do
        echo 'layer_size '$layer_size' layer_depth '$layer_depth
        
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/dubins_bilinear.py \
        --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
        --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
        --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
        --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &

        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python ood_similar_shifts/dubins_bilinear.py --fourier \
        --bc-hidden-depth $layer_depth --bc-hidden-layer-size $layer_size \
        --deep-sets-hidden-depth $layer_depth --deep-sets-hidden-layer-size $layer_size \
        --NN-hidden-depth $layer_depth --NN-hidden-layer-size $layer_size \
        --bilinear-hidden-depth $layer_depth --bilinear-hidden-layer-size $layer_size &
    done
done