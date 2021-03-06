% Recovery of block sparse signals using the pattern-clustered sparse
% Bayesian learning (PC-SBL) method
% Nov. 1 2013, written by Yanning Shen

clear;close all;
img=imread('lena256.bmp');
% img=mean(img,3);%for color image
ss=[128,128]; % size of image
X0=imresize(img,ss);
X0=double(X0);
X=X0;
[a,b]=size(X);

% Discrete Wavelet Transform
load DWTM.mat
M=64;
R=randn(M,a);
SNR=120;
measure=R*X;
% Observation noise
stdnoise = std(measure)*10^(-SNR/20);
noise = randn(M,1) * stdnoise;

Y=measure+noise;

A=R*ww';

figure(1);
X=reshape(X,ss);
imshow(uint8(X));
title('Original Image')




%=========================================================
 %                Proposed PC-SBL algorithm
 %=========================================================

X3=zeros(a,b);
eta=1;
for i=1:128
    rec1=PCSBL(Y(:,i),A,stdnoise(i),eta); % recover the image column by column
    X2(:,i)=rec1;
end

figure(2);
X2=ww'*X2;  %  inverse-DWT transform
X22=reshape(X2,ss);
ERR=sqrt(sum(sum((X22-X0).^2,1),2)/sum(sum(X.^2,1),2))
imshow(uint8(X22));
title('PC-SBL');


