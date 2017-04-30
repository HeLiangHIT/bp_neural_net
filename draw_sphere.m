function draw_sphere(pos,r,color)
%% 根据三维坐标pos向量在空间中绘制球体

% pos = 20*rand(10,3);%各个位置
% r=2;%球体半径

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