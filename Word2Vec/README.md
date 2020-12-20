# bert部署

韩露露 2020/12/18

## 1.BERT介绍

BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder，因为decoder是不能获要预测的信息的。模型的主要创新点都在pre-train方法上，即用了Masked LM和Next Sentence Prediction两种方法分别捕捉词语和句子级别的representation。

原理介绍：https://zhuanlan.zhihu.com/p/46652512

2018/11谷歌放出官方代码和预训练模型，包括 BERT 模型的 TensorFlow 实现、BERT-Base 和 BERT-Large 预训练模型和论文中重要实验的 TensorFlow 代码。

谷歌开源地址：https://github.com/google-research/bert

## 2.bert-as-service安装

bert-as-service参考地址：https://bert-as-service.readthedocs.io/en/latest/

**安装要求：python>=3.5,1.10=<tensorflow<=2.0.0**

### Step1 安装bert-as-service

```
pip install -U bert-serving-server bert-serving-client
```

### Step2 下载预训练模型

模型地址：https://github.com/google-research/bert#pre-trained-models

![image-20201220142918469](C:\Users\HanLulu\AppData\Roaming\Typora\typora-user-images\image-20201220142918469.png)

下载`BERT-Base，Uncased`,并解压到脚本的同一级目录内，文件夹名应该为`uncased_L-12_H-768_A-12`，不是请手动更改！

### Step3 运行脚本BERTServer.py

![image-20201220143808699](C:\Users\HanLulu\AppData\Roaming\Typora\typora-user-images\image-20201220143808699.png)

**请保证服务一直出于开启状态！**

### Step4 运行脚本Word2Vec.py

服务端接收到客户端需求：

![image-20201220144012200](C:\Users\HanLulu\AppData\Roaming\Typora\typora-user-images\image-20201220144012200.png)

等待客户运行出结果即可，具体时间看CPU性能！



