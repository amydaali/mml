function D = distMat( X, Y )
% function D = distMat( X, Y )
% Computes the pair-wise Euclidian distance between two sets of
% observations.
%
% usage
%     D = distMat( X, Y )
%
% input
%     X: (N,M)-matrix where each row is an observation.
%     Y: (K,M)-matrix where each row is an observation.
%
% output
%     D: The pair-wise squared Euclidian distance between each of the 
%     vectors in X with each of the vectors in Y. The entry D(i,j) 
%     corresponds to the distance between the i-th row in X and 
%     the j-th row in Y.
%
% description
%     Computes the pair-wise Euclidian distance between two sets of
%     observations, X and Y, where each observation is a row vector. 
%     
%     On a side-note: this function is around 15-0% faster per run than 
%     the Ian Nabey version. It is also about 3-18% faster on small 
%     matrices with few entries and around 100-200% slower for matrices 
%     with many entries compared to Matlab's pdist2 function. The dimension
%     of the observation does not seem to matter much.
%     Note that this is not benchmarked extensively! And should probably use 
%			pdist2 since its way faster...
%
%
% author
%     Martin Hjelm, mar.hjelm@gmail.com

[N,Dim1] = size(X);
[K,Dim2] = size(Y);

if Dim1 ~= Dim2
	error('The row vectors must be of the same dimension!')
end

% Compute: || x - y ||^2 = dot(x,x) + dot(y,y) - 2*dot(x,y)
D = ((sum(X.*X,2))*ones(1,K)) + (ones(N,1)*(sum(Y.*Y,2)')) - ((2.*X)*(Y'));

% Set any negative values in D, due to rounding errors, to zero.
D(D<0) = 0;

end