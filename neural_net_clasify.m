function y = neural_net_clasify(net,x)
%% 根据net计算向量x的输出y以及各层的输出layer_out
sigmoid = @(x) (1./(1+exp(-x)));%函数声明

y = [];
for p =1:size(x,1)
    layer_out{1} = x(p,:)';%X必须是一个一维向量
    for k=2:net.lNum
        %依次计算各层的输出
        w = net.w{k-1};%前一层->当前层的权值向量
        layer_out{k} = sigmoid(w*layer_out{k-1});%当前层输出
    end
    y = [y;layer_out{k}'];
end

end