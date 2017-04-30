function y = neural_net_clasify(net,x)
%% ����net��������x�����y�Լ���������layer_out
sigmoid = @(x) (1./(1+exp(-x)));%��������

y = [];
for p =1:size(x,1)
    layer_out{1} = x(p,:)';%X������һ��һά����
    for k=2:net.lNum
        %���μ����������
        w = net.w{k-1};%ǰһ��->��ǰ���Ȩֵ����
        layer_out{k} = sigmoid(w*layer_out{k-1});%��ǰ�����
    end
    y = [y;layer_out{k}'];
end

end