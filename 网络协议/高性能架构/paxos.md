# 分布式一致性算法Paxos
### 前提简介：


### 目的
paxos算法主要解决的问题就是如何保证分布式系统中各个节点都能执行一个相同的操作序列。在2f+1个节点的集群中，它可以允许有f个节点是不可用的。

paxos是在多个成员之间对某个==提议==达成一致性的算法。这个提议可以是很多东西，例如选取leader。

#### 角色
在paxos中主要有三个角色，分别为:提案者（proposer）、接受者（acceptor）、学习者（learner）.

- 提案者（proposer）:proposer的工作在于接收客户端请求，将其封装成提案（proposal）。并将提案（proposal）发给所有的接受者（acceptor）。根据接受者（acceptor）返回情况，控制是否需要提交该提案（proposal）即保存该提案（proposal）.
- 接收者（accepter）：接收者主要是对参与对提案的投票，接收和处理paxos两个阶段的请求。
- 学习者（learner）：学习者不参与提案和投票，只被动接收提案结果。他的存在可用于扩展读性能，或者跨区域之间读操作。（**并非paxos协议的核心组成部分**）同步给其他未确定的accepter

> 注：提案是指：指需要达成共识的某一个值，或者某一个操作。paxos对其封装成一个提案，并为其生成唯一的提案编号。本文中使用M, V表示一个提案，其M表示提案编号，V表示需要达成共识的值。

#### prepare阶段
prepare阶段，由提案者向接收者发送提案的prepare请求，接受者根据约定决定是否需要响应该请求。如果接受者通过提案M, 的准备请求，则向提案者保证以下承诺

- 接受者承诺不再通过==编号小于等于M的提案==的==prepare请求==
- 接受者承诺不再通过编号小于M的提案的**accepter请求**，也就是不再通过编号小于M的提案
- 如果acceptor已经通过某一提案，则承诺在prepare请求的响应中返回已经通过的==最大编号的提案内容==。如果没有通过任何提案，则在prepare请求的响应中返回空值
其中prepare阶段还得注意，在prepare请求中，proposer只会发提案编号，也就是M。

#### accept阶段
accept阶段，提案者如果在prepare阶段收到大多数响应后，由提案者向接受者发送accept请求。例如此时进行决策的提案是M, V，根据接受者在prepare阶段对提案者的承诺来进行决策。

- 如果此时acceptor没有通过编号大于M的prepare请求，则会批准提案M, V，并返回已通过的编号最大的提案（也就是M, ）。
- 如果此时acceptor已经通过编号大于M的prepare请求，则会拒绝提案M, V，并返回已通过的编号最大的提案（大于M的编号）。


提案者会统计收到的accept请求的响应，如果响应中的编号等于自己发出的编号，则认为该acceptor批准过了该提案。如果存在大多数acceptor批准了该提案，则记作该提案已达成共识如果没有大多数acceptor批准该提案，则重新回到prepare阶段进行协商。

其中accept阶段也有注意的地方，在prepare请求中，**proposer只会发提案M,** 。而在accept请求，**proposer会发送提案编号和提案值，也就是M, V。**这里要注意的是V的值，如果在prepare请求的响应中，部分acceptor已经批准过的提案值，则V为prepare请求的响应中编号最大的提案值，否则可以由proposer任意指定。

#### learn阶段
learn阶段，在某一个提案通过paxos达成共识之后，由acceptor通知learner学习提案结果。

#### 具体算法流程
1. 先为proposal生成一个编号n，这里需要保证编号全局唯一，并且全局递增，具体实现全局编号，这里不予讨论。
2. proposer向所有acceptors广播prepare(n)请求
3. acceptor比较n和minProposal，如果n>minProposal则执行minProposal=n，并且将 acceptedProposal 和 acceptedValue 返回给proposer。
4. proposer接收到过半数回复后，如果发现有acceptedValue返回，将所有回复中acceptedProposal最大的acceptedValue作为本次提案的value，否则可以任意决定本次提案的value。
5. 到这里可以进入第二阶段，广播accept (n,value) 到所有节点。
4. acceptor比较n和minProposal，如果n>=minProposal，则acceptedProposal=minProposal=n，acceptedValue=value，本地持久化后，返回；否则，返回minProposal。
5. proposer接收到过半数请求后，如果发现有返回值result（minProposal） > n，表示有更新的提议，跳转到1；否则value达成一致。