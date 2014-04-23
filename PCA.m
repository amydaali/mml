function [Z, V, D] = PCA ( X, m, scaleOn )
% function [Z, U, D] = PCA ( X, m, scaleOn )
% PCA - Principal Component Analysis
%
% usage
% [Z, U, D] = PCA ( X, m )
%
% input
% 	X : (M x N)-matrix of row vectors
% 	m : number of vectors that that is choosen to span the sub-space of X
%
% output
% 	Z : projection of X onto the PC subspace
% 	U : main components / eigenvectors of the PCA analysis or basis vectors of the sub space if you so whish
% 	D : main component values / eigenvalues of the PCA analysis
%
% description
% 	Computes the PCA of the provided column vector matrix X.
%
% author
%     Martin Hjelm, mar.hjelm@gmail.com
%
% copyright
%     Do what ever you want but give me credit, if credit is due.

%%%%%%%%%% CHECK INPUT ETC. %%%%%%%%%%%%

% Check erroneous input
  if nargin < 1
      error('PCA.m: Too few input arguments. For help type help PCA.\n');    
  end

  [M,N] = size ( X );
  
  if ~(exist('m','var'))
    m = N;
  end
  
  if ~(exist('scaleOn','var'))
    scaleOn = 0;
  end  
  
  if M*N == 0
      error('PCA.m: Empty input matrix not allowed. For help type help PCA.\n');
  end
  if m <= 0
      error('PCA.m: Number of principal componenets to retain m must be positive. For help type help PCA.\n');
  end


%{
%%%%%%%%%% DO (OLD) PCA ALGORITHM %%%%%%%%%%%%

% 1. Mean is calculated in the matlab cov function so skip step

% 2. Compute the scatter matrix (which is proportional against the covariance)
  S = X*X' / sqrt( N-1 );

% 3. Compute the eigenvectors and eigenvalues of S
  [U,D] = eig ( S );

% 4. Get the m largest eigenvalues with corresponding eigenvectors

	% 4.1 Sort the eigenvalues and vectors, largest first
	  [U,D] = sorteigen ( U, D );	  
    D = max(D);

	% 4.2 Choose the m biggest
	  U = U(:,1:m);
	  D = D(:,1:m);

% 5. Calculate the projection of X on to the PC m-subspace. 
  Z = X * U(:,1:m);


%%%%%%%%%% END %%%%%%%%%%%%
%}
  
  
  

%%%%%%%%%% DO (BETTER SVD BASED) PCA ALGORITHM %%%%%%%%%%%%

%{
The covariance  
X'*X is hermitian... so  
  X'*X = W*D*W'
where W is the eigen vectors and D the diagonal eigen value matrix. 
We have svd(X) = USV' so  
  X'*X =  (USV')' * USV' = VS'U' * USV' = {U*U' = I} = V(S'S)V' 
so 
  D = S*S' and W = V;
%}

% 1. Remove mean
X = X - ones(M,1) * mean(X,1);

% 1.5 If scaling is on, scale each dimensions variance to sum to one.
if scaleOn
  X = X ./ (ones(M,1)*std(X,[],1));
end

% 2. Compute the SVD (we get already sorted eigen values and eigen vectors!)
% D - eigenvalues and V eigen vectors 
[~,S,V] = svd( X / sqrt(N-1) );
% [U,S,V] = svd ( X / sqrt(N-1), 'econ');

% 3. Choose the m biggest
V = V(:,1:m);
S = diag(S'*S);
D = S(1:m);

% 4. Calculate the projection of X on to the PC m-subspace in the biggest variance direction. 
Z = X * V;


%%%%%%%%%% END %%%%%%%%%%%%


end
