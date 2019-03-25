# FEM

所谓智能材料的组成性反应，例如：磁流变弹性体，由它们的微观结构（组成和分布）决定。通常，这些杰出材料的显着特征是通过混合具有特定物理性质的细颗粒（可磁化铁颗粒）和便于使用标准技术处理或加工的基质材料来设计。


PM：众多小Pm的体积平均值 macroscopic Piola-Kirchhoff stress Pm is the volume average of Pm 
Pm：microscopic counterpart Pm
FM：宏观形变的梯度，定义为FEM有限元仿真的边界条件；

betat:一个点位置向量x 与 空间b一一对应
是一对非线性关系：x = Y(X), X=y(x)

F:形变梯度，F=Grad Y, Y是形变映射
YY(X)：表示标量磁势，
HH：拉格朗日磁场
HH = Grad YY(X)

A：磁矢量势

保护线性动量
DivP = 0 in B0

DivBB = 0 in B0

调查聚类方法结合神经网络架构作为运行中间模拟的潜在替代。


结构可靠度求解的功能函数是由结构的基本随
机变量(如结构荷载和结构参数等)和响应量(如应
力、位移等)构成的，通过结构有限元分析，可以得到
结构的响应量以及随机变量和响应量的映射关系。
但有限元计算过程复杂，通过其获得足够多的数据
来求解结构可靠度，由于计算量大而显然不可取。
然而，通过结构有限元有限而又足够次数的分析计
算，却可为神经网络的学习与检验提供足够的数据
样本，为神经网络训练提供了充足的训练和检验样


# 运行

- 环境
    + tensorflow1.1.0
    + 
- 指令
    + python CNN_FEM.py
    + python RNN_FEM.py
    + python NN_FEM.py
    + tensorboard --logdir=logs
