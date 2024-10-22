我添加了 `augmentation.py`，里面有我进行数据增强的实现

我主要修改 `main.py` 的两部分，实现 augmentation 和 wandb report

对于 augmentation

```python
if augmentation:
    X_val, y_val = X_train[20000:], y_train[20000:]
    X_train, y_train = X_train[:20000], y_train[:20000]

    X_train_aug, _, y_train_aug, _ = load_cifar_2d('../augmented_cifar-10_data')
    # X_train_aug, _, y_train_aug, _ = load_cifar_2d('../cifar-10_data')
    
    X_val_aug, y_val_aug = X_train_aug[20000:], y_train_aug[20000:]
    X_train_aug, y_train_aug = X_train_aug[:20000], y_train_aug[:20000]

    X_train = np.concatenate((X_train, X_train_aug), axis=0)
    y_train = np.concatenate((y_train, y_train_aug), axis=0)

if augmentation:
    X_train_aug, _, y_train_aug, _ = load_cifar_2d('./cifar-10_data')
    X_train = np.concatenate((X_train, X_train_aug), axis=0)
    y_train = np.concatenate((y_train, y_train_aug), axis=0)
```

```python
if use_wandb:
	wandb.init(
		project="ANN-HW2",
		# name="mlp",
  		# name="mlp-aug-self",
		name="mlp-aug",
  		# name="mlp-wobn",
    	# name="mlp-l-r-d-b-l",
	)

if use_wandb:
    wandb.log({"train_loss": loss_.cpu().data.numpy(), "train_acc": acc_.cpu().data.numpy()})

if use_wandb:
    if test:
        wandb.log({"test_loss": loss_.cpu().data.numpy(), "test_acc": acc_.cpu().data.numpy()})
    else:
        wandb.log({"valid_loss": loss_.cpu().data.numpy(), "valid_acc": acc_.cpu().data.numpy()})
return acc, loss
```