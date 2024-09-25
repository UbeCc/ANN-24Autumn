#!/bin/bash

# loss=("hinge")
# ac=("selu" "hardswish" "tanh")

# loss=("kl" "softmax" "hinge" "focal")
loss=("focal")
ac=("selu")

for l in "${loss[@]}"
do
  for a in "${ac[@]}"
  do
    echo "loss: $l, ac: $a"
    python run_mlp.py --loss $l --activation $a &
  done
done