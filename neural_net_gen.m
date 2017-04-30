function net = neural_net_gen(lNum,pNum,varargin)
%% 产生神经网络net，参数：
% lNum 表示层数，数值
% pNum 表示各层的节点数目，一维向量
% net 输出的net结构体
% 其它输入：
% minErr 误差界
% maxIter 最多迭代次数
% enta 步长，通常0-1之间
% alpha 惯性系数，0-1之间


net.lNum = lNum;%层数
net.pNum = pNum;%各层的节点数目
net.ErrIter = inf;%误差
if length(varargin) == 4
    net.minErr = varargin{1};%误差界
    net.maxIter = varargin{2};%最多迭代次数
    net.enta = varargin{3};%步长，通常0-1之间
    net.alpha  = varargin{4};%惯性系数，0-1之间
else
    % 默认参数
    net.minErr = 0.1;%误差界
    net.maxIter = 300;%最多迭代次数
    net.enta = 0.5;%步长，通常0-1之间
    net.alpha  = 0.5;%惯性系数，0-1之间
end

%初始化各个节点权值
for k=1:(lNum-1)
    w{k} = rand(pNum(k+1),pNum(k));%各层权值，本层节点数x上一层节点数，这样安排有利于输出相乘
    deltaw{k} = zeros(pNum(k+1),pNum(k)); %初始化该变量用于后面的计算需要
end
net.w=w;%输出神经网络的权向量
net.deltaw = deltaw;%缓存的上一步w变换量
end