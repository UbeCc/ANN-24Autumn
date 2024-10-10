# ANN HW2 CNN

����Ȼ ��23 2022010229

2024.10.10

---

## `self.training` ����

�� `mlp` ��ʵ��Ϊ����

BatchNorm1d �е� self.training��

```python
def forward(self, input):
    if self.training:
        mean = input.mean(dim=0)
        var = input.var(dim=0, unbiased=False)
        
        with torch.no_grad():
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
    else:
        mean = self.running_mean
        var = self.running_var

    x_normalized = (input - mean) / torch.sqrt(var + self.eps)
    return self.weight * x_normalized + self.bias
```

�� `BatchNorm1d` �У�`self.training` ��������ʹ�õ�ǰ���ε�ͳ�����ݣ���ֵ�ͷ������ʹ������ʱ�ۻ���ͳ�����ݡ���ѵ��ʱ������ϣ��ģ����Ӧÿ�����ε����ݷֲ������ڲ���ʱ����ϣ��ʹ������ѵ�������ȶ�ͳ�����ݡ�

Dropout �е� self.training��

```python
def forward(self, input):
    if self.training:
        mask = torch.bernoulli(torch.full_like(input, 1 - self.p))
        return input * mask / (1 - self.p)
    else:
        return input
```

�� `Dropout` �У�`self.training` �������Ƿ�Ӧ�� `dropout`����ѵ��ģʽ�£������������Ԫ�������ǵ�ֵ��Ϊ0���Է�ֹ����ϡ��ڲ���ģʽ�£�������Ԫ�����ֻ�Ծ�������ǲ���Ҫ�����κ����ţ���Ϊ��ѵ��ʱ�Ѿ�ͨ������ (1 - p) ���в�����

�ۺ���˵��

���� BatchNorm��
- ѵ��ʱʹ������ͳ����������ģ����Ӧ���ݵı仯��
- ����ʱʹ��ȫ��ͳ������ȷ��Ԥ���һ���ԣ����ܵ������ε�Ӱ�졣

���� Dropout��
- ѵ��ʱ���������Ԫ�����ڷ�ֹ����ϣ���ǿģ�͵ķ���������
- ����ʱ����������Ԫ���Գ������ģ�͵�������������Ԥ�⡣

## `baseline` ���

`mlp` ����

- learning_rate: 1e-3
- batch_size: 100
- epochs: 30
- drop_rate: 0.25
- eps: 1e-5
- momentum (�����ƶ�ָ��): 0.1

![alt text](assets/mlp.png)

`cnn` ����

- learning_rate: 1e-2
- batch_size: 100
- epochs: 20
- drop_rate: 0.2
- eps: 1e-5
- momentum (�����ƶ�ָ��): 0.1

![alt text](assets/cnn.png)

## ѵ��������֤�� `loss` �Ƚ�

### `loss` ��ͬԭ��

1. �����

ѵ����ʧ����ģ���Ѿ�ѧϰ���������ϼ���ģ�ģ�Ϳ��ܿ�ʼ����ѵ�������е��ض�ģʽ������ѵ����ʧ�ϵͣ���֤��ʧ����ģ��δ�����������ϼ���ģ����ģ�͹���ϣ�������Щδ�����������ϱ��ֻ���������֤��ʧ�ϸߡ�

2. ����

����dropout��L2���򻯵ȼ���ͨ����ѵ��ʱ�Ǽ���ģ�������֤ʱ���������ܵ���ѵ����ʧ����û������ʱ�����������֤��ʧ����ֱ���ܵ�Ӱ�졣

3. ������ȫ������

ѵ����ʧͨ������һ��epoch�ж��mini-batch��ƽ��ֵ����֤��ʧͨ������������֤����һ���Լ���ġ�

### ���ε���

1. ������ϣ�

���ѵ����ʧ�����½�����֤��ʧ��ʼ�������������ԵĹ�����źš��������Ҫ�������򻯣�����ģ�͸��Ӷȣ����ռ�����ѵ�����ݡ�

2. ѧϰ�ʵ�����

���������ʧ�����½���ѵ����ʧ���Ը��ͣ���������ѧϰ�ʣ�������߶����ȶ��½��ҽӽ���ѧϰ�ʿ����Ѿ��ܿ����ˡ�

3. ��ͣ��

ѵ����ʧ����֤��ʧ֮��Ĳ����԰���ȷ����ʱֹͣѵ���Է�ֹ����ϡ�

## ���Լ����

����ڶ�����
> `baseline` ���

���У�

`mlp` ����Ч����`0.52`
`CNN` ����Ч����`0.69`

## ȥ�� `BN`

`mlp` ȥ�� `BN`

![alt text](assets/mlp-wobn.png)

`cnn` ȥ�� `BN`

![alt text](assets/cnn-wobn.png)

�����������У����� `BN` ���ᵼ�����ղ���׼ȷ�������½������������Ϊ������һ�����ܽ����ݱ�׼����һ����׼�ֲ���ʹѵ�����̸����ȶ��Ϳ��٣�������ģ�͸��õķ���������

�Լ������ǿ��Կ�������CNN�����У�������ʹ��������һ��ʱ����֤��ʧ��׼ȷ�ʵ����߱�ò�ƽ��������ζ����֤���̲��ȶ������������Ϊ����֤�����У����ڹ�һ���ľ�ֵ��ʹ��ѵ�������е����о�ֵ����ġ����ֵ�����뵱ǰ��֤���ݵľ�ֵ��ͬ������������������ƫ�ơ������ϣ�����ѵ�����̵Ľ��У����о�ֵ����ӽ��������ݼ��ľ�ֵ��ƫ�ƻ��С��

## ȥ�� `dropout`

`mlp` ȥ�� `dropout`

![alt text](assets/mlp-wodropout.png)

`cnn` ȥ�� `dropout`

![alt text](assets/cnn-wodropout.png)

�����������У����� `dropout` ��ᵼ�����صĹ���ϡ���ѵ�����������㹻��ʱ����֤��ʧ��ʼ�������� `MLP` �У���֤׼ȷ���� 20 �� epoch ֮ǰ�ʹﵽ�����ֵ����Ҳ�����˹���ϡ���ˣ����ǿ��Եó����ۣ�`dropout` ����Է�ֹ����ϡ�

## Bonus1 ����˳��

����Ҫ���и����ļܹ����ҽ����� `BN`��`dropout`��`Relu` �� `Linear` ���˳�򣬾���ܹ����ͼע������ `b->BN`, `d->dropout`, `r->Relu`, `l->Linear`

![alt text](assets/mlp-order.png)

![alt text](assets/cnn-order.png)

���ʵ���������ǿ���֪��

### BN -> Dropout -> ReLU
BN �� Dropout ֮ǰ���ܻᵼ��ÿ�� mini-batch ��ͳ�����ݲ�һ�£���Ϊ Dropout ���������һЩ��Ԫ��
Dropout �� ReLU ֮ǰ���ܻᶪ��һЩ��Ҫ��������

### Dropout -> BN -> ReLU
Dropout �� BN ֮ǰ���ܻ�Ӱ�� BN ��ͳ�����ݼ��㡣
BN ���ܻᲿ�ֵ��� Dropout ������Ч����

### ReLU -> Dropout -> BN
����˳��ᵼ�� BN ����յ�������ֲ��仯�ϴ�

## Bonus2 ������ǿ

```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # clip
    transforms.RandomHorizontalFlip(),     # flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # color shift
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 norm parmeter
])
```

�Ҷ�ѵ�����ݽ�����������ǿ����������¡�Ҫע����� CIFAR-10 ��һ���Լ��Ĺ�һ��ϵ��...

![alt text](assets/mlp-aug.png)

���룬�������£�
- ����ѵ����������ͨ�����������ݽ��б任���������ݼ���ģ��������������ݶ���һ�����Ч����ģ�ͳ��⣩���������������޵�����ر����ã����԰���ģ��ѧϰ������������仯��
- ���ģ�ͷ�����������ǿ������ݰ����˸���ı仯�����������԰���ģ��ѧϰ��³��������������ģ�Ͷ��ض����������Ĺ����������������ʵ���糡���еı��֡�
- ģ����ʵ����ı仯����ʵ��Ӧ���У��������ݿ��ܻ��и��ֱ仯������ա��Ƕȡ�λ�õȣ���������ǿ���԰���ģ����Ӧ��Щ�仯�����ʵ��Ӧ���е����ܡ�
- ƽ�����ݼ����������ƽ������ݼ�������ͨ������������и������ǿ��ƽ�����������������