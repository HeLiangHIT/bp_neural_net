function net = neural_net_gen(lNum,pNum,varargin)
%% ����������net��������
% lNum ��ʾ��������ֵ
% pNum ��ʾ����Ľڵ���Ŀ��һά����
% net �����net�ṹ��
% �������룺
% minErr ����
% maxIter ����������
% enta ������ͨ��0-1֮��
% alpha ����ϵ����0-1֮��


net.lNum = lNum;%����
net.pNum = pNum;%����Ľڵ���Ŀ
net.ErrIter = inf;%���
if length(varargin) == 4
    net.minErr = varargin{1};%����
    net.maxIter = varargin{2};%����������
    net.enta = varargin{3};%������ͨ��0-1֮��
    net.alpha  = varargin{4};%����ϵ����0-1֮��
else
    % Ĭ�ϲ���
    net.minErr = 0.1;%����
    net.maxIter = 300;%����������
    net.enta = 0.5;%������ͨ��0-1֮��
    net.alpha  = 0.5;%����ϵ����0-1֮��
end

%��ʼ�������ڵ�Ȩֵ
for k=1:(lNum-1)
    w{k} = rand(pNum(k+1),pNum(k));%����Ȩֵ������ڵ���x��һ��ڵ�������������������������
    deltaw{k} = zeros(pNum(k+1),pNum(k)); %��ʼ���ñ������ں���ļ�����Ҫ
end
net.w=w;%����������Ȩ����
net.deltaw = deltaw;%�������һ��w�任��
end