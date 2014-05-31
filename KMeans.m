function [Mu, r] = KMeans ( X, k, max_iter, varargin ) 
% function [Mu, r] = KMeans ( X, k, max_iter, prog_fun ) 
% Kmeans - K-means Clustering Algorithm
%
% usage
%     [Mu, r] = KMeans ( X, k, max_iter, prog_fun ) 
%
% input
%        X : (m,n)-matrix of column vectors
%        k : number of cluster centers
%        max_iter : maximum number of iterations
%        prog_fun : I dunno....  
%
% output
%     Mu : The k cluster centers the algorithm produces.
%     r : Cluster labels for each data point
%
% description
%     Kmeans 
%
% author
%     Martin Hjelm, martinhjelm@kth.se

  % Check erroneous input
  if nargin < 2
    error('KMeans.m: Too few input arguments. For help type help KMeans.\n');    
  end

  % Check if dist2 function is present. 
  if ~exist('dist2')
    error('KMeans requires dist2 function by Ian Nabey')
  end

  % Set default values for the number of iterations, if not set.
  if ~(exist('max_iter','var'))
    max_iter = 100;
  end
  
  if numel(varargin) > 0
    Mu = varargin{1};
    clusterCentersSet = 1;
  else 
    clusterCentersSet = 0;
  end
  

  %  1. Choose k random cluster centers and iniate storage matrices
  [~, N] = size ( X );
  if ~clusterCentersSet
    Mu = X ( :, randi ( N, k, 1, 'uint32' ) ); % cluster  matrix
  end
  % vector
  r  = zeros ( N, 1 );  % index vector  

  %  2. Do iteration up till max_iter or till no change is observable
  for i_iter = 1:max_iter
    %disp(['Iteration ',num2str(i),' of ',num2str(max_iter)]); drawnow;

    % Find new nearest cluster center for each point
    [~,r_new] = min(dist2(X', Mu'),[],2);
    
    % Compute new cluster centre for each by taking the mean of every point from
    % given the new assigments.
    for j = 1:k
      Mu(:,j) = mean ( X(:, ( r_new == j ) ), 2 );     
    end
    
    % Check if cluster assignment has changed, break if not.
    if r == r_new
      %disp([num2str(k),' iteration. Nothing changed so break loop..']); drawnow;
      break;      
    end

    %disp(['Number of vectors changed in this iteration: ',num2str(sum(r==r_new))]); disp(' '); drawnow;

    % Update r vector
    r = r_new;

    % If function handle exists do feval - data visualization program
%     if exist('prog_fun','var')
%       feval ( prog_fun, X, Mu, r );
%       pause 
%     end

  end
  
  
%disp(' ');disp('End of function');disp(' --------------------------------');disp(' ');
end











