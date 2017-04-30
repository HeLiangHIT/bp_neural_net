
clear all; clc; close all;
%% 训练数据初始化
vecLen=3;%特征维度
sampleN=10;%各类样本数据个数
raw_data = [1.58	2.32	-5.8	0.21	0.03	-2.21	-1.54	1.17	0.64;%第一类3特征，第二类3特征，第三类3特征
    0.67	1.58	-4.78	0.37	0.28	-1.8	5.41	3.45	-1.33;
    1.04	1.01	-3.63	0.18	1.22	0.16	1.55	0.99	2.69;
    -1.49	2.18	-3.39	-0.24	0.93	-1.01	1.86	3.19	1.51;
    -0.41	1.21	-4.73	-1.18	0.39	-0.39	1.68	1.79	-0.87;
    1.39	3.16	2.87	0.74	0.96	-1.16	3.51	-0.22	-1.39;
    1.20	1.40	-1.89	-0.38	1.94	-0.48	1.40	-0.44	0.92;
    -0.92	1.44	-3.22	0.02	0.72	-0.17	0.44	0.83	1.97;
    0.45	1.33	-4.38	0.44	1.31	-0.14	0.25	0.68	-0.99;
    -0.76	0.84	-1.96	0.46	1.49	0.68	-0.66	-0.45	0.08]';
Xin=reshape(raw_data(:),vecLen,[])';%分类1、2、3、1、2、3...
Yd=repmat(eye(3),sampleN,1);%理想输出
Yd_ind = repmat([1:3]',sampleN,1);%理想输出
%绘制数据
% figure('Name','样本数据')
% draw_sphere(Xin,0.3,Yd);%绘制样本
% scatter3(Xin(1:3:end,1),Xin(1:3:end,2),Xin(1:3:end,3),'MarkerFaceColor',Yd(1,:));hold on
% scatter3(Xin(2:3:end,1),Xin(2:3:end,2),Xin(2:3:end,3),'MarkerFaceColor',Yd(2,:));
% scatter3(Xin(3:3:end,1),Xin(3:3:end,2),Xin(3:3:end,3),'MarkerFaceColor',Yd(3,:));
% 原始数据本来就无法收敛，因此误差无论如何都是比较大的
%% 训练样本整理
keep = 1:size(Xin,1); keep([16]) = [];%去除错误影响较大的数据
Xin_fix = Xin(keep,:); Yd_fix = Yd(keep,:);

%% 产生神经网络
net = neural_net_gen(3,[vecLen,20,3],...%产生和初始化神经网络，3层，各层分别为3、10、1个节点
    0.05,100000,0.5,0.9);%设定各个迭代控制参数
%此处可以直接修改net的属性用于迭代

%% 训练网络
net1 = neural_net_train_per(net,Xin_fix,Yd_fix);
net2 = neural_net_train_mass(net,Xin_fix,Yd_fix);
%% 识别样本
yo1 = neural_net_clasify(net1,Xin);%逐个样本训练
[~,yo1] = max(yo1,[],2);
per1 = sum(yo1(:) ~=Yd_ind(:))/length(Yd_ind(:));
yo2 = neural_net_clasify(net2,Xin);%成批样本训练
[~,yo2] = max(yo2,[],2);
per2 = sum(yo2(:) ~=Yd_ind(:))/length(Yd_ind(:));
[per1,per2]


figure('Name','误差曲线');
plot(net1.ErrIter,'b.-'),hold on; grid on
plot(net2.ErrIter,'rx-');
ylim([min([net1.ErrIter(end),net2.ErrIter(end)]),max([net1.ErrIter(1),net2.ErrIter(1)])]);
legend('逐个样本修正','成批样本修正');xlabel('迭代次数');ylabel('均方误差');
