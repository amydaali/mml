function [Z, U, D] = pca ( X, m )
% function [Z, U, D] = PCA ( X, m )
% PCA - Principal Component Analysis
%
% usage
% [Z, U, D] = PCA ( X, m )
%
% input
% 	X : (M x N)-matrix of column vectors
% 	m : number of vectors that that is choosen to span the sub-room of X
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
  if nargin < 2
      error('PCA.m: Too few input arguments. For help type help PCA.\n');    
  end

  [M,N] = size ( X );
  
  if M*N == 0
      error('PCA.m: Empty input matrix not allowed. For help type help PCA.\n');
  elseif m <= 0
      error('PCA.m: Number of principal componenets to retain m must be positive. For help type help PCA.\n');
  end


%{
%%%%%%%%%% DO (OLD) PCA ALGORITHM %%%%%%%%%%%%

% 1. Mean is calculated in the matlab cov function so skip step

% 2. Compute the scatter matrix (which is proportional against the covariance)
  S = X*X' / ( N-1 );

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

% XX' is diagonal so XX' = WDW' where W is the eigen vectors and D the
% diagonal eigen value magtrix. 
% But svd(X) = USV' but XX' = USV' * (USV')' = USV' VSU' = {V*V' = I} =
% US.^2U' so D = S.^2 and W = U;

% 1. Remove mean
X = X - mean(X,2) * ones(1,N);

% 2. Compute the SVD (we get already sorted eigen values and eigen vectors!)
% D - eigenvalues and U eigen vectors 
[~,D,U] = svd ( X' / sqrt(N-1), 0);

% 3. Choose the m biggest
U = U(:,1:m);
D = D(:,1:m);

% 4. Calculate the projection of X on to the PC m-subspace in the biggest variance direction. 
Z = U' * X;


%%%%%%%%%% END %%%%%%%%%%%%


end
