# 基于MNIST数据集的AutoEncoder


## README

- 使用简单的mlp模型实现；
- net训练代码详见train.py；
- encoded的可视化、decoded的可视化实现详见Visualization.py；
- 在Visualization.py中，跳过了训练环节，使用net_params.pkl中的参数。
- 2020.10.26增加了generate.py，将训练完成的autoencoder进行图像的生成。


## RESULTS

```p'y
Epoch: 1        Training Loss: 5.32389379
Epoch: 2        Training Loss: 5.05373490
Epoch: 3        Training Loss: 4.64951849
Epoch: 4        Training Loss: 4.13836312
Epoch: 5        Training Loss: 4.01902878
Epoch: 6        Training Loss: 3.97812617
Epoch: 7        Training Loss: 4.13038373
Epoch: 8        Training Loss: 3.60661268
Epoch: 9        Training Loss: 4.02023005
Epoch: 10       Training Loss: 3.77055931
Epoch: 11       Training Loss: 3.83698082
Epoch: 12       Training Loss: 3.73578644
Epoch: 13       Training Loss: 3.56306863
Epoch: 14       Training Loss: 3.74601460
Epoch: 15       Training Loss: 3.81022811
Epoch: 16       Training Loss: 3.69482625
Epoch: 17       Training Loss: 3.75265217
Epoch: 18       Training Loss: 3.52381110
Epoch: 19       Training Loss: 3.69612586
Epoch: 20       Training Loss: 3.69877911
Epoch: 21       Training Loss: 3.78072667
Epoch: 22       Training Loss: 3.63295662
Epoch: 23       Training Loss: 3.72138798
Epoch: 24       Training Loss: 3.60215092
Epoch: 25       Training Loss: 3.53074872
Epoch: 26       Training Loss: 3.77659965
Epoch: 27       Training Loss: 3.83549023
Epoch: 28       Training Loss: 3.80836987
Epoch: 29       Training Loss: 3.55092967
Epoch: 30       Training Loss: 3.37368929
```



## VISUALIZATION

### Encoded
training data被编码为2维向量的分布：
![encoded](https://github.com/linhaowei1/Learning-Learning/blob/main/AutoEncoder/pic/encoded.png)



### Decoded
training data前后经过encoder的前后对比图
![decoded](https://github.com/linhaowei1/Learning-Learning/blob/main/AutoEncoder/pic/decoded.png)

### 解码图片生成
经过decoder，可以将两个特征的向量生成一张28*28的图片。
下面利用（x，y） 在 （-5,5）； （-10, 10）； （-50, 50）三个范围内的随机数各进行了100张图片的生成。

#### 范围【-5：5】
![1](https://github.com/linhaowei1/Learning-Learning/blob/main/AutoEncoder/pic/-5,5.png)
#### 范围【-10：10】
![1](https://github.com/linhaowei1/Learning-Learning/blob/main/AutoEncoder/pic/-10,10.png)
#### 范围【-50：50】
![1](https://github.com/linhaowei1/Learning-Learning/blob/main/AutoEncoder/pic/-50,50.png)

