#  python3 一些常用的好用的方法，常可以用来刷题

在遇到链表的题目时候，有时候可以采用**伪头节点**来方便我们做题。

有时候要求我们要有实现随机删除的话，那么如果我们每次**删除的都是最后一个元素**，那么就可以在O（1）的时间内实现删除操作！
### 字符串去除空格

假如S是一个字符串，那么如下：
```
S.strip(): 去除字符串两边的空格
S.lstrip():去除字符串左边的空格
S.rstrio():去除字符串右边的空格

print(str.upper())          # 把所有字符中的小写字母转换成大写字母
print(str.lower())          # 把所有字符中的大写字母转换成小写字母
print(str.capitalize())     # 把第一个字母转化为大写字母，其余小写
print(str.title())          # 把每个单词的第一个字母转化为大写，其余小写
```

python中ASCII码字符与int之间的转换：主要使用ord方法跟chr方法
```
ASCII码转换为int：ord('A')    65

int转为ASCII码：chr(65)   'A'
```

python的bin()函数可以返回一个int类型正数的二进制表示
int()函数的可以将一个字符/字符串转成整型。全称是int（x,base=10）base表示进制


python中使用队列
```
#导入容器，这是一个队列  定义的这是一个双向队列
from  collections import deque 
append():默认在右边插入一个元素
appendleft(): 在左边加入一个元素
clear(): 清空队列
pop()： 删除一个元素，默认右边
popleft(): 删除最左边的元素
reverse(): 逆置元素
remove(value)： 移除第一次value的元素
```

inf 表示无穷大的意思：例如
float("inf"):正无穷大
float("-inf"):负无穷大

除此之外，还要再看看python的集合库:
```
from sortedcontainers import SortedSet
# 这是实现有序集合的一种方式
```
python获取某个元素在list的下标。例如：a元素
```
下标= list.index(a)
list.remove(index) # 删除值为index的元素
list.pop(index) # 删除index位置的元素

```

## python中使用堆
数据进行装入
```
import heapq as hq   ## 导入对应的库

heap = []           ## 定义新生成的堆
#使用heapq库的heappush函数将数据堆入
for i in data:        ##data是待排序的元素
     hq.heappush(heap,i)  
```

### 排列组合
```
comb(n, k):#组合函数 从n个数中选k个数的选法
```

#### 计数
```
collections.Counter(list):统计list中出现的数量。返回成一个类似dist的字典
```
### 注意一些求幂的操作


##### 字符串跟表达式互换
```
eval函数将字符串当成有效Python表达式来求值，并返回计算结果
eval(str) -> int


与之对应的repr函数，它能够将Python的变量和表达式转换为字符串表示
```


#### 有序字典的使用
```
data = [('a',1),('b',3),('c',2)]
#按数据中key值的大小排序
od = collections.OrderedDict(sorted(data,key=lambda s:s[0]))
print(od)
#按数据中value值的大小排序
od = collections.OrderedDict(sorted(data,key=lambda s:s[1]))
print(od)

如果是key=lambda s:(-s[1],s[0]):表示先按照1号位置排序 然后按照2号位置排序，其中-表示逆序

lambda x: (-len(x), x)):其中(-len(x), x))指首先用x的长度排序，如果长度相同则用出现的先后排序。

intervals = sorted(intervals ,key = lambda x: x[0])
指按照intervals中每个元素的第一个值的大小进行排序

dict.item():会返回一些遍历的元组对象
```

#### python 中使用堆
直接导入heapq就可以
这个模块提供了堆队列的实现，也称为优先级队列
这个模块可以实现小根堆。它使用了==数组==来实现：从零开始计数，对于所有的 k ，都有 heap[k] <= heap[2*k+1] 和 heap[k] <= heap[2*k+2] 
我们可以使用以下函数来获取对应的数据：
```
heapq.heappush(heap, item):将 item 的值加入 heap 中，保持堆的不变性.item 可以是一个数组。

heapq.heappop(heap)： 弹出并返回heap中最小的元素
heapq.heappushpop(heap, item)：将 item 放入堆中，然后弹出并返回 heap 的最小元素。该组合操作比先调用 heappush() 再调用 heappop() 运行起来更有效率。

heapq.heapify(x) 将list x 转换成堆，原地，线性时间内。

heapq.merge(*iterables, key=None, reverse=False)将多个已排序的输入合并为一个已排序的输出（例如，合并来自多个日志文件的带时间戳的条目）。 返回已排序值的 iterator。

```
1. 堆元素（item）可以为元组。这适用于将比较值（例如任务优先级）与跟踪的主记录进行赋值的场合。
2. 堆排序 可以通过将所有值推入堆中然后每次弹出一个最小值项来实现。这类似于 sorted(iterable)，但与 sorted() 不同的是这个实现是不稳定的。
