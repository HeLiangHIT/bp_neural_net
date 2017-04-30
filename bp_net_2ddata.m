
clear all; clc; close all;
%% ѵ�����ݳ�ʼ��
vecLen=2;%����ά��
sampleN=30;%�����������ݸ���
[Xin,Yd] = sample_gen_2d(sampleN);

%% ����������
net = neural_net_gen(3,[vecLen,4,1],...%�����ͳ�ʼ�������磬3�㣬����ֱ�Ϊ3��10��1���ڵ�
    1e-3,500*sampleN,0.1,0.0);%�趨�����������Ʋ���
%�˴�����ֱ���޸�net���������ڵ���

%% ѵ������
net1 = neural_net_train_per(net,Xin,Yd);%�������ѵ��
net2 = neural_net_train_mass(net,Xin,Yd);%��������ѵ��
%% ʶ������
yo1 = neural_net_clasify(net1,Xin);%�������ѵ��
yo1 = yo1>=0.5;
per1 = sum(yo1 ~=Yd)/length(Yd);
yo2 = neural_net_clasify(net2,Xin);%��������ѵ��
yo2 = yo2>=0.5;
per2 = sum(yo2 ~=Yd)/length(Yd);
[per1,per2]


figure('Name','�������');
plot(net1.ErrIter,'b.-'),hold on; grid on
plot(net2.ErrIter,'rx-');
ylim([min([net1.ErrIter(end),net2.ErrIter(end)]),max([net1.ErrIter(1),net2.ErrIter(1)])]);
legend('�����������','������������');xlabel('��������');ylabel('�������');
