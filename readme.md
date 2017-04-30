**模式识别课程实验**
参考实验报告的原理介绍。


**函数和脚本介绍**

__函数：__
draw_sphere.m 绘制训练样本空间分布的函数
neural_net_clasify.m 使用训练好的网络对输入样本分类
neural_net_gen.m 根据输入的网络参数产生并初始化网络
neural_net_train_mass.m 成批样本修正的BP算法
neural_net_train_per.m 逐个样本修正的BP算法
sample_gen_2d.m 产生二维测试样本并绘制其分布

__脚本：__
bp_net_2ddata.m 使用二维测试样本测试网络性能
bp_net_rawdata.m 使用实验数据进行网络性能测试

__运行说明：__
bp_net_2ddata.m 运行该脚本运行可以产生图3-2和图3-3，但是由于测试样本数随机产生的，因此不能保证曲线完全一直。
bp_net_rawdata.m 运行该脚本可以产生第四节的图，根据实验说明调节脚本中语句net = neural_net_gen(...)函数的输入参数来产生不同实验结果。

>__注：__
本程序支持多层网络，各层节点数量可以任意设置，已经封装完毕，各个函数的功能参考其注释，实现原理和分析参考实验报告。不排除有没有考虑到的错误，如果发现bug将非常感谢您联系作者改正，另外也期待您的建议：helianghit@foxmail.com