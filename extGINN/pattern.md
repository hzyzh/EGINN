### extGINN接受的输入（.json）文件格式

基本框架如下：

```json
{
 "callers":[...],
 "target":{...}, 
 "callees":[...]
}
```

其中callers中存储所有目标函数的caller的信息，target中存储目标函数的信息，callees中存储目标函数的所有callee的信息

首先通过spoon-interval构建出的CFG，找出所有调用了目标函数的函数（caller），以及目标函数调用了的所有函数（callee），只引入一层的context。



#### callers部分

callers中依次存储了目标函数的每个caller；其中每个caller的基本架构与原GINN接受的json文件格式基本相同：划分为若干个interval，对应的CFG称为i-CFG。caller的interval内部则额外增加calling_mask部分，用于指示调用目标函数的节点位置信息。如下所示：

```json
"callers":[
    {
        "0":{
            ...,
            "calling_mask":[0,1,0,...]
        },
        "1":{
            ...,
            "calling_mask":[...]
        },
        "numOfNode":...,
        "graph":[[...],...]
    }
    {...},
    ...
]
```

其中calling_mask为一个长度为n的0/1数组（n为该caller的该i-CFG中节点个数），若第i位为1，表示第i个节点调用了目标函数



#### target部分

target部分则存储了目标函数相关的所有信息，也划分为若干个interval，interval内部额外增加callee_masks部分；interval外部的结构与原json文件相同。如下所示：

```json
"target":{
    "0":{
        ...,
        "callee_masks":[[0,1,0,...],[...],...,[...]]
    },
	"1":{
        ...,
        "callee_masks":[[...],...]
    },
	...
	"numOfNode":...,
	"bugPos":...,
	...
}
```

callee_masks部分为一个$k\times V_i$的0/1数组，$k$为callee的个数，$V_i$为目标函数第i个interval内的节点个数。若第i个interval的callee_masks的(p, q)位为1，则表示第i个interval的第q个节点调用了callee$_p$



#### callees部分

callees部分就是依次存储了所有被目标函数调用到的callee信息，除了没有calling_mask之外，结构与callers部分完全相同：

```json
"callees":[
    {
        "0":{...},
        "1":{...},
        ...
        "numOfNode":...,
        "graph":[[...],...]
    }
    {...},
    ...
]
```



#### 流程

extGINN framework的流程大致为：

1. 在每个caller的i-CFG上分别进行GINN的message passing，根据calling_mask筛选出所有调用了目标函数的节点，记录这些节点的embeddings，通过一个MLP layer学习得到目标函数的context信息，作为目标函数的entry node的输入message

2. 在目标函数的interval-CFG图上进行GINN的message passing

3. 根据callee_masks分别筛选出目标函数中调用各个callee的所有节点，分别记录下这些节点的信息，汇总后分别作为各个callee的entry node的输入message；也可省略此步，则对callee不再引入context信息

4. 在每个callee的i-CFG上分别进行GINN的message passing，并将各个callee分别学习得到一个综合embedding，最后汇总得到如下形式的callee embedding vector

   ```json
   [[0,0,0,...,0], [...embedding for callee_1...], ..., [...embedding for callee_n...]]
   ```

5. 根据callee_masks以及callee embedding vector来获得目标函数中各个节点对应的callee context，学习得到更新的目标函数embeddings

6. 重复2 ~ 5步若干次