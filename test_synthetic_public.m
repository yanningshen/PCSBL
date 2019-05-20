% Recovery of block sparse signals using the pattern-clustered sparse
% Bayesian learning (PC-SBL) method
% Nov. 1 2013, written by Yanning Shen and Jun Fang

%% Initialization
clear; 
n=100;                                          % signal dimension
m=40;                                           % number of measurements
K=20;                                           % total number of nonzero coefficients
L=3;                                            % number of nonzero blocks

%SNR = 100;    % Signal-to-noise ratio
sigma2=1e-12; 

 % generate the block-sparse signal 
x=zeros(n,1);
r=abs(randn(L,1)); r=r+1; r=round(r*K/sum(r)); 
r(L)=K-sum(r(1:L-1));                           % number of non-zero coefficients in each block
g=round(r*n/K);
g(L)=n-sum(g(1:L-1));
g_cum=cumsum(g);
    
for i=1:L
    % generate i-th block 
    seg=rand(r(i),1);                % generate the non-zero block
    loc=randperm(g(i)-r(i));        % the starting position of non-zero block
    x_tmp=zeros(g(i), 1);
    x_tmp(loc(1):loc(1)-1+r(i))= seg; 
    x(g_cum(i)-g(i)+1:g_cum(i), 1)=x_tmp;
end    

% generate the measurement matrix
Phi=randn(m,n);
A=Phi./(ones(m,1)*sqrt(sum(Phi.^2)));

% noiseless measurements
measure=A*x;

% Observation noise, stdnoise = std(measure)*10^(-SNR/20);
stdnoise=sqrt(sigma2);
noise=randn(m,1)*stdnoise;

% Noisy measurements
y=measure+noise;
 
    
 %% Revoery via PC-SBL   

 % Initialization of parameters  
a=0.5;
b=1e-10;
c=1e-10;
d=1e-10;
eta=1;

iter=0;
iter_mx=100;
D=eye(n);
sigma2=1;
alpha_new=ones(n,1);
var_new=inv(A'*A/sigma2+D);
mu_old=ones(n,1);
mu_new=1/sigma2*var_new*A'*y;
gamma_new=1/sigma2;
while iter<iter_mx& norm(mu_new-mu_old)>1e-6
    iter=iter+1;
    mu_old=mu_new;
    mul=[mu_new(2:n);0];
    mur=[0;mu_new(1:n-1)];
    var=diag(var_new);
    varl=[var(2:n);0];
    varr=[0;var(1:n-1)];
    E=mu_new.^2+eta*mul.^2+eta*mur.^2+var+eta*varl+eta*varr;
    alpha_new=a./(0.5*E+b);
    idx1=find(alpha_new>1e10);
    alpha_new(idx1)=1e10;
    alf=[alpha_new(2:n); 0];                                %   left-shifted version
    arf=[0; alpha_new(1:n-1)];                              %   right-shifted version
    D=diag(alpha_new+eta*alf+eta*arf);
    %=============================================
    %  estimate the variance of noise
     num=(y-A*mu_old)'*(y-A*mu_old)+trace(var_new*A'*A)+2*d;
    den=m+2*c;
    sigma2=num/den;
    %==============================================
    var_new=inv(A'*A/sigma2+D);
    mu_new=1/sigma2*var_new*A'*y;
end
x_new=mu_new;
mse=norm(x_new-x)^2