## EGINN demo

demo.py程序会训练并测试一个简单的EGINN模型，需要用户将训练数据和测试数据通过参数传递给程序。

训练数据传递格式形如：

```bash
--train TRAIN1 TRAIN2 TRAIN3 ...
```

其中TRAIN均为训练数据(.json文件)的路径，且至少要接收一个参数

测试数据传递格式形如：

```bash
--test TEST1 TEST2 TEST3 ...
```

其中TEST均为测试数据(.json文件)的路径。--test为可选参数，若不提供，则默认使用训练集进行测试

#### 数据

所有数据（.json文件）都储存在./jsondata/目录下。**每次更换训练或测试数据时，都需要将 ../data 内的所有文件删除**！

#### 示例

输入指令：

```shell
rm -rf ../data
python3 ./demo.py --train ./jsondata/input.json
```

