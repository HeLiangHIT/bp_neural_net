function [Xin,Yd] = sample_gen_2d(N)
% 产生二维的测试样本和理想输出

% N=50;%样本数
%% 产生绘制样本
%第一类
u1 = [-2 -2]';
sigma1 = [1 0;0 1];
r1 = mvnrnd(u1,sigma1,N);
%第二类
u2 = [2 2]';
sigma2 = [1 0;0 2];
r2 = mvnrnd(u2,sigma2,N);

figure('Name','样本数据')
plot(r1(:,1),r1(:,2),'ro');
hold on;
plot(r2(:,1),r2(:,2),'bp');
legend('第一类','第二类')

Xin = [r1;r2];%前面R1类，后面R2类
Yd = [zeros(N,1);ones(N,1)];%前面R1类0，后面R2类1

end
