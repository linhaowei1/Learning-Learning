# textCNN实验记录

## cpu

在测试集上的acc为$79.00\%$。跑20个epoch。

```python
hyperparameters = {
    'embed_dim' : 300,
    'sent_len' : 15,
    'in_channel' : 1,
    'out_channel' : 100,
    'ker_size1' : 3,
    'ker_size2' : 4,
    'ker_size3' : 5,
    'drop_rate' : 0.5,
    'class_num' : 4,
    'weight_decay' : 1e-2,
    'hidden_dim' : 300,
    'batch_size' : 128,
    'lr' : 1e-3
}
```

实验数据：

```python
| training | epoch 1/20 | train loss 0.0101 
| evaluation | valid loss 0.0091 | Acc   0.6360
 
| training | epoch 2/20 | train loss 0.0083 
| evaluation | valid loss 0.0084 | Acc   0.7180
 
| training | epoch 3/20 | train loss 0.0076 
| evaluation | valid loss 0.0082 | Acc   0.7240
 
| training | epoch 4/20 | train loss 0.0072 
| evaluation | valid loss 0.0081 | Acc   0.7460
 
| training | epoch 5/20 | train loss 0.0070 
| evaluation | valid loss 0.0082 | Acc   0.7320
 
| training | epoch 6/20 | train loss 0.0070 
| evaluation | valid loss 0.0081 | Acc   0.7390
 
| training | epoch 7/20 | train loss 0.0068 
| evaluation | valid loss 0.0080 | Acc   0.7540
 
| training | epoch 8/20 | train loss 0.0068 
| evaluation | valid loss 0.0079 | Acc   0.7560
 
| training | epoch 9/20 | train loss 0.0067 
| evaluation | valid loss 0.0080 | Acc   0.7580
 
| training | epoch 10/20 | train loss 0.0067 
| evaluation | valid loss 0.0079 | Acc   0.7800
 
| training | epoch 11/20 | train loss 0.0067 
| evaluation | valid loss 0.0080 | Acc   0.7660
 
| training | epoch 12/20 | train loss 0.0067 
| evaluation | valid loss 0.0080 | Acc   0.7590
 
| training | epoch 13/20 | train loss 0.0066 
| evaluation | valid loss 0.0079 | Acc   0.7740
 
| training | epoch 14/20 | train loss 0.0066 
| evaluation | valid loss 0.0080 | Acc   0.7720
 
| training | epoch 15/20 | train loss 0.0066 
| evaluation | valid loss 0.0079 | Acc   0.7690
 
| training | epoch 16/20 | train loss 0.0067 
| evaluation | valid loss 0.0079 | Acc   0.7800
 
| training | epoch 17/20 | train loss 0.0066 
| evaluation | valid loss 0.0080 | Acc   0.7760
 
| training | epoch 18/20 | train loss 0.0067 
| evaluation | valid loss 0.0079 | Acc   0.7600
 
| training | epoch 19/20 | train loss 0.0068 
| evaluation | valid loss 0.0079 | Acc   0.7710
 
| training | epoch 20/20 | train loss 0.0067 
| evaluation | valid loss 0.0079 | Acc   0.7720
```

## Gpu

在测试集上的acc为84.90%。跑100个epoch，网络结构更大，参数更多。

超参数：

```python
hyperparameters = {
    'embed_dim' : 1000,
    'sent_len' : 15,
    'in_channel' : 1,
    'out_channel' : 600,
    'ker_size1' : 3,
    'ker_size2' : 4,
    'ker_size3' : 5,
    'drop_rate' : 0.5,
    'class_num' : 4,
    'weight_decay' : 1e-2,
    'hidden_dim' : 300,
    'batch_size' : 512,
    'lr' : 1e-3
}
```

训练过程：

```python
| training | epoch 90/100 | train loss 0.0016 
| evaluation | valid loss 0.0019 | Acc   0.8330
 
| training | epoch 91/100 | train loss 0.0016 
| evaluation | valid loss 0.0019 | Acc   0.8320
 
| training | epoch 92/100 | train loss 0.0016 
| evaluation | valid loss 0.0019 | Acc   0.8250
 
| training | epoch 93/100 | train loss 0.0016 
| evaluation | valid loss 0.0019 | Acc   0.8320
 
| training | epoch 94/100 | train loss 0.0016 
| evaluation | valid loss 0.0019 | Acc   0.8350
 
| training | epoch 95/100 | train loss 0.0016 
| evaluation | valid loss 0.0019 | Acc   0.8310
 
| training | epoch 96/100 | train loss 0.0016 
| evaluation | valid loss 0.0019 | Acc   0.8290
 
| training | epoch 97/100 | train loss 0.0016 
| evaluation | valid loss 0.0019 | Acc   0.8290
 
| training | epoch 98/100 | train loss 0.0016 
| evaluation | valid loss 0.0019 | Acc   0.8250
 
| training | epoch 99/100 | train loss 0.0016 
| evaluation | valid loss 0.0019 | Acc   0.8360
 
| training | epoch 100/100 | train loss 0.0016 
| evaluation | valid loss 0.0019 | Acc   0.8300
```

