Implementation of EWC for MNIST by using Chainer
===

This is an implementation of Elastic Weight Consolidation (EWC) introduced in [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796).


- 3 layer NN
- hidden num = 50
- lambda = 1
- epoch = 100


Task A only
---
<img src="https://github.com/okdshin/EWC/blob/master/example_result/train_task_a.png?raw=true" width="600">

Continuously training for Task B (without EWC)
---
<img src="https://github.com/okdshin/EWC/blob/master/example_result/train_task_ab.png?raw=true" width="600">

Continuously training for Task B (with EWC)
---
<img src="https://github.com/okdshin/EWC/blob/master/example_result/train_task_ab_with_ewc.png?raw=true" width="600">
