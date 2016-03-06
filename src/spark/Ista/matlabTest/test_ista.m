clear;

vfile = '../data/v_4x6x3_1/v_4x6x3_1_0';
dfile = '../data/d';
xfile = '../data/x';
yfile = '../data/y';


gamma = 0.01;
lambda = 0.1;
step = 50;

Vs = dlmread(vfile);
V = sparse(Vs(:,1)+1,Vs(:,2)+1,Vs(:,3));

D = dlmread(dfile);

x = dlmread(xfile);
y = dlmread(yfile);



for i=1:step
    yh = D*V*x;
    norm(y-yh)^2
    gx=V'*D'*(yh-y);
    xn = wthresh(x-gamma*gx,'s',lambda);
    norm(x - xn)^2
    x = xn;
end

