# generate-lyrics-using-PyTorch (神经网络生成歌词)
use RNN to generate chinese lyrics

使用循环神经网络实现了歌词的生成，RNN采用了GRU来实现

## Environment (运行环境)
```
Linux
Python3.6
Pytorch
```

## Example 
```
input> 情人
length> 120
gen> 情人的飞翔
如果我遇见世界的梦想
那为了永远
永远不会看不到
如果你看我的幸福
你的温柔像羽毛
巷口上的画面
一天看你看清楚的名字就像日历上一件
没有你可以爱你的微笑
洋溢幸福的味道
还原谅了我的情人节
你是我唯一的爱情
只有这生变我的爱
```

## Recommend reading
* [Pytorch docs](http://pytorch.org/docs)
* [Chinese Pytorch docs](http://pytorch-cn.readthedocs.io/)
* [PyTorch Tutorials](http://pytorch.org/tutorials/)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## usage
*  project structure
```
|- project
    |- data
    |- runtime
    |- config.py        
    |- dataset.py       
    |- model.py
    |- utils.py
    |- train.py         
    |- test.py
```
* adjust parameters
```
config.py is the configuration file ， you can adjust parameters in the file
```

* train model
```
python3 train.py
```

* evaluate model
```
python3 test.py
```

## 实现过程请阅读我的博客

[我的CSDN博客](http://blog.csdn.net/m0_37687051/article/details/77675844)


## 问题反馈
欢迎和我交流！共同学习！

* 邮件(lxm_0828#163.com, 把#换成@)
* QQ: 929325776
* weibo: [@捏明](http://weibo.com/littlelxm)

## 感激
感谢以下的项目,排名不分先后

* [Generating Names with a Character-Level RNN](http://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html) 
* 数据借用了此项目的数据 [机器学习PAI为你自动写歌词，妈妈再也不用担心我的freestyle了](https://yq.aliyun.com/articles/134287)


## 关于作者

```
学生一枚，就读于山大（shanxi university）CS专业
对deep learning(深度学习), NLP(自然语言处理)，NLU(自然语言理解)，Big Data(大数据)有狂热的学习欲望
精通WEB开发
```
