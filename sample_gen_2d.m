function [Xin,Yd] = sample_gen_2d(N)
% ������ά�Ĳ����������������

% N=50;%������
%% ������������
%��һ��
u1 = [-2 -2]';
sigma1 = [1 0;0 1];
r1 = mvnrnd(u1,sigma1,N);
%�ڶ���
u2 = [2 2]';
sigma2 = [1 0;0 2];
r2 = mvnrnd(u2,sigma2,N);

figure('Name','��������')
plot(r1(:,1),r1(:,2),'ro');
hold on;
plot(r2(:,1),r2(:,2),'bp');
legend('��һ��','�ڶ���')

Xin = [r1;r2];%ǰ��R1�࣬����R2��
Yd = [zeros(N,1);ones(N,1)];%ǰ��R1��0������R2��1

end
