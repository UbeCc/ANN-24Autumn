修改如下
1. 在 run_mlp.py 中，用 argparse 替换掉原始硬编码参数
2. 在 grid_run_mlp.py 中，修改 grid 搜索逻辑
3. 在 loss.py 中，增加检查逻辑
4. 在 solve_net.py 中，添加 report 函数，并修改 train_net 和 test_net，便于向 wandb 报告
5. 在 batch_eval.sh 中，添加多函数实验训练逻辑
6. 在 grid_search_wandb.yaml, run_grid_search.sh 中，添加超参搜索逻辑
7. 在 draw.py 中，添加画图逻辑