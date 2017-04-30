function draw_sphere(pos,r,color)
%% ������ά����pos�����ڿռ��л�������

% pos = 20*rand(10,3);%����λ��
% r=2;%����뾶

[x,y,z]=sphere(3);
for k=1:size(pos,1)
    xt = r*x+pos(k,1);
    yt=  r*y+pos(k,2);
    zt=  r*z+pos(k,3);
    surf(xt,yt,zt,'FaceColor',color(k,:),'EdgeColor',color(k,:)*0.5); 
    hold on;
end

axis equal;
axis tight;
% colormap('autumn');

end