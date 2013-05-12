function negLL = mvnLike ( X, mu, covar, varargin )
  % function negLL = mvnLike ( X, mu, covar )
  %
  % usage
  % negLL = mvnLike ( X, mu, covar, varargin )
  %
  % input
  % 	X: (M x N)-matrix of row vectors i.e. M datapoints in N dimensions
  %   mu: (1 x N) row vector - the mean vector for the mvn distribution
  %   covar: (N x N ) matrix - the covariance matrix for the mnv distribution.
  %   varargin: Cholesky decompostion of the covariance matrix in case of
  %   multiple runs for example i.e. we do not have to recalculate it all the
  %   time.
  %
  % output
  % 	negLL: The multivariate Normal negative log likelihood.  
  %
  % description
  % 	Calculates the negative multivariate Normal log likelihood.
  %
  % author
  %     Martin Hjelm, mar.hjelm@gmail.com
  %
  % copyright
  %     Do what ever you want but give me credit, if credit is due.
  %

  
  % The multi-variate negative log likelihood for dataset where every row
  % is a sample and every column is the dimension
  [M,N] = size(X);
  
  % Check args 
  if nargin < 3
    error('Not enough input arguments')
  end
  
  % Check that X, mean and cov are consistent
  if size(mu,2) ~= N
    error('Mean vector has a different dimension than the data vector.');
  end
  
  if size(covar,2) ~= N
    error('Covariance has a different dimension than the data vector.');
  end
  
  if N == 1 % If one dimensonal we assume we got the variance not the covariance.
    xBar = X - mu;
    negLL = M * 0.5 * ( log(2) + log(pi) + log( covar^2 ) ) + ( 1 / ( 2 * covar^2 ) ) * ( xBar' * xBar );
    return
  end
  
  xBar = X - ones(M,1) * mu;
  % Use a bit of cholesky decomposition to do faster calculations for
  % Mahalanlobis distance and log det of covariance   
  % Check if decomposition is provided. 
  if nargin == 4
    L = varargin{1};
  else
    L = chol ( covar );  
  end
  logCovDet = 2 * sum( log ( diag(L) ) );
  sqrdMahalan = xBar * ( L \ ( L' \  xBar' ) );  
  negLL =  M * 0.5 * ( N * ( log(2) + log(pi) ) + logCovDet ) + sum( 0.5 * diag(sqrdMahalan) );

end


