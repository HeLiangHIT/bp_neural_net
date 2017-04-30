function net = neural_net_train_mass(net,Xin,Yd)
%% ��������ѵ����������������ڵ�Ȩֵ����������������������BP�㷨��������
% net �����磬Ȩֵ����ǰ��
% Xin ����������[������x����ά��] �ľ���
% Yd �����������

%% ������ʼ��
% ��������
minErr = net.minErr;%����
maxIter = net.maxIter;%����������

% �����������
lNum = net.lNum;%����
pNum = net.pNum;%����Ľڵ���Ŀ

% ��������
ErrIter = [];%����¼
sampleN = size(Xin,1);%������������

%% BP�㷨����������������
Errtmp = inf;%��ʼ���
for k = 1:maxIter%*sampleN%���Ա�֤ѭ�����
    p = mod(k-1,sampleN)+1;%��ǰ���������
    %% ��ǰ�����������Ԫ���ֱ�����һ��
    xin = Xin(p,:)';%�����������
    yd = Yd(p,:)';%�����������
    [yo,layer_out{p}] = neural_net_clasify(net, xin);%����������������ĸ������
    
    %% ��ֹ�ж�
    yerr = yd - yo; %���շ������
    if p == 1 %���ָտ�ʼ
        Errtmp = Errtmp/sampleN;%����ֵ
        ErrIter = [ErrIter,Errtmp];%������һ�������
        if Errtmp<minErr
            break;%��ֹ�������
        end
        Errtmp = sum(yerr.^2);
    else%������������ѭ��
        Errtmp = Errtmp + sum(yerr.^2);%�ۼ����
    end
    
    %% �Ӻ���ǰ�������μ���������Ȩֵ����������
    err_out{p} = neural_net_err(net,layer_out{p},yd);
    if p == sampleN %���һ�֣���������Ȩֵ
        net.w = neural_net_adjust(net,layer_out,err_out);
    end
    
    
end
net.ErrIter = ErrIter;%����������ߣ����ڻ��������ٶȶԱ�

fprintf('��ֹ���=%1.4e,  ��������=%d\n',ErrIter(end),k);
end

function [y,layer_out] = neural_net_clasify(net,x)
%% ����net��������x�����y�Լ���������layer_out
sigmoid = @(x) (1./(1+exp(-x)));%��������

layer_out{1} = x;%X������һ��һά����
for k=2:net.lNum
    %���μ����������
    w = net.w{k-1};%ǰһ��->��ǰ���Ȩֵ����
    layer_out{k} = sigmoid(w*layer_out{k-1});%��ǰ�����
end
y = layer_out{net.lNum};
end


function err_out = neural_net_err(net,layer_out,yd)
%% ����������
N = length(layer_out);%����

%% ��������������
err_out{N} = -layer_out{N}.*(1-layer_out{N}).*(yd - layer_out{N});%�����
for k=(N-1):-1:1
    errTmp = net.w{k}'*err_out{k+1} ;%��һ������ӳ�䵽��ǰ������ڵ�
    err_out{k} = layer_out{k}.*(1-layer_out{k}).*errTmp;%���Ϲ�ʽ
end

end


function wadj = neural_net_adjust(net,layer_out,err_out)
%% �����������У��Ȩֵ
enta = net.enta;%������ͨ��0-1֮��
alpha  = net.alpha;%����ϵ����0-1֮��
sampleN = length(layer_out);%��������
N=length(layer_out{1});%����

%% Ȩֵ����
for m=(N-1):-1:1%�Ը���
    deltaw = 0;%��ʼ��
    for n=1:sampleN%�Ը����������
        deltaw = deltaw -enta*err_out{n}{m+1}*layer_out{n}{m}';
    end
    net.deltaw{m} = deltaw + net.deltaw{m}*alpha;%��һ���Ĺ���Ҳ������
    wadj{m} = net.w{m} + net.deltaw{m};%Ȩֵ����
end

end