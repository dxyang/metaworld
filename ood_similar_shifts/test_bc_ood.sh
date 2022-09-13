# CUDA_VISIBLE_DEVICES=1 python ood_similar_shifts/test_bc.py --task-name basketball-v2 --split-per 0.5 --obj-ood --ood-axis 0
CUDA_VISIBLE_DEVICES=1 python ood_similar_shifts/test_bc.py --task-name basketball-v2 --split-per 0.5 --goal-ood --ood-axis 0 > output/basketball_goal.txt
# #
CUDA_VISIBLE_DEVICES=0 python ood_similar_shifts/test_bc.py --task-name button-press-v2 --split-per 0.5 --obj-ood --ood-axis 0 > output/button_obj.txt
# #
# CUDA_VISIBLE_DEVICES=1 python ood_similar_shifts/test_bc.py --task-name dial-turn-v2 --split-per 0.5 --obj-ood --ood-axis 0
# CUDA_VISIBLE_DEVICES=1 python ood_similar_shifts/test_bc.py --task-name dial-turn-v2 --split-per 0.5 --goal-ood --ood-axis 0
# #
CUDA_VISIBLE_DEVICES=0 python ood_similar_shifts/test_bc.py --task-name drawer-close-v2 --split-per 0.5 --obj-ood --ood-axis 0 > output/drawer_obj.txt
# #
CUDA_VISIBLE_DEVICES=0 python ood_similar_shifts/test_bc.py --task-name peg-insert-side-v2 --split-per 0.5 --obj-ood --ood-axis 0 > output/peg-insert-side_obj.txt
CUDA_VISIBLE_DEVICES=0 python ood_similar_shifts/test_bc.py --task-name peg-insert-side-v2 --split-per 0.5 --goal-ood --ood-axis 1 > output/peg-insert-side_goal.txt
#
CUDA_VISIBLE_DEVICES=0 python ood_similar_shifts/test_bc.py --task-name pick-place-v2 --split-per 0.5 --obj-ood --ood-axis 0 > output/pick-place_obj.txt
# CUDA_VISIBLE_DEVICES=1 python ood_similar_shifts/test_bc.py --task-name pick-place-v2 --split-per 0.5 --goal-ood --ood-axis 0
CUDA_VISIBLE_DEVICES=0 python ood_similar_shifts/test_bc.py --task-name pick-place-v2 --split-per 0.2 --goal-ood --ood-axis 0 > output/pick-place_goal.txt
#
CUDA_VISIBLE_DEVICES=0 python ood_similar_shifts/test_bc.py --task-name push-v2 --split-per 0.5 --obj-ood --ood-axis 0 --res-grasp-state > output/push_obj.txt
# CUDA_VISIBLE_DEVICES=5 python ood_similar_shifts/test_bc.py --task-name push-v2 --split-per 0.5 --goal-ood --ood-axis 0
CUDA_VISIBLE_DEVICES=0 python ood_similar_shifts/test_bc.py --task-name push-v2 --split-per 0.2 --goal-ood --ood-axis 0 > output/push_goal.txt
#
# CUDA_VISIBLE_DEVICES=4 python ood_similar_shifts/test_bc.py --task-name reach-v2 --split-per 0.5 --goal-ood --ood-axis 0
# CUDA_VISIBLE_DEVICES=4 python ood_similar_shifts/test_bc.py --task-name reach-v2 --split-per 0.5 --goal-ood --ood-axis 0 --train-bc --expert-break-on-succ
CUDA_VISIBLE_DEVICES=5 python ood_similar_shifts/test_bc.py --task-name reach-v2 --split-per 0.2 --goal-ood --ood-axis 0 > output/reach_goal.txt
#
# CUDA_VISIBLE_DEVICES=5 python ood_similar_shifts/test_bc.py --task-name sweep-into-v2 --split-per 0.5 --obj-ood --ood-axis 0
#
# CUDA_VISIBLE_DEVICES=5 python ood_similar_shifts/test_bc.py --task-name window-open-v2 --split-per 0.5 --obj-ood --ood-axis 0
# CUDA_VISIBLE_DEVICES=0 python ood_similar_shifts/test_bc.py --task-name window-open-v2 --split-per 0.2 --obj-ood --ood-axis 0


#DEBUG
# CUDA_VISIBLE_DEVICES=0 python ood_similar_shifts/test_bc.py --task-name reach-v2 --split-per 0.2 --goal-ood --ood-axis 0 --debug --res-timestep --res-full-state
CUDA_VISIBLE_DEVICES=0 python ood_similar_shifts/test_bc.py --task-name reach-v2 --goal-ood --expert-break-on-succ --res-timestep --render
# CUDA_VISIBLE_DEVICES=0 python ood_similar_shifts/test_bc.py --task-name push-v2 --split-per 0.5 --obj-ood --ood-axis 0 --debug --res-timestep --res-grasp-state

CUDA_VISIBLE_DEVICES=0 python ood_similar_shifts/test_bc.py --task-name reach-v2 --goal-ood --res-timestep --reduced-state-space --eval-res-best --debug --train-bc


CUDA_VISIBLE_DEVICES=4 python ood_similar_shifts/reach_notebook_script.py
CUDA_VISIBLE_DEVICES=0 python ood_similar_shifts/push_notebook_script.py

