function [ C, opt_param ] = cv2 ( X, y, classifier_handle, params, N_folds, N_repetitions, varargin )
% function [ C, opt_param ] = cv ( X, y, classifier_handle, { param name, value range}, nfolds, nrepetitions, loss_function )
% K-fold cross-validation with repitition for classifier. 
%
% usage
%   [ C, opt param ] = cv ( X, y, classifier_handle, { param_name, value_range}, nfolds, nrepetitions, loss_function )
%
% input
%   X: (m,n)-matrix of row vectors of data.
%   y: (1,n)-vector that contains the labels.
%   classifier_handle: The handle for the classifier that gets called with the 
%                      parameters.
%   params: Cell arrray containing all the parameter arguments with 
%           structure {para_name1, param_value1, param_name2,...} i.e.
%           {'pcaDim',[20, 25, 100, 200, 500, 0]}
%   nfolds: Optional - The number of partitions, default is 10.
%   nrepetitions: Optional - The number of repititions, default is 5.
%   loss_function: Optional - Name of the loss function, default is the 
%                  zero_one_loss function at end of file. Your function 
%                  must take the input true values and predicted values.   
%
% output
%   C: The classifier trained with the best parameters
%   opt_param: A structure containing the optimal parameters.
%
% description
%   Function for running crossvalidation for a  classifier over a number of
%   parameter combinations. The function tests every parameter combination and then
%   returns the best combination based upon the error estimate.
%     
%   Example:
%   [ C, opt_param ] = cv ( X, y, @mySVNfun, ...
%   {'param1',[20, 25, 100, 200, 500, 0],'param2',[234, 4, 444, 65],},...
%   5, 10, @myLossFun )
%
% author
%     Martin Hjelm, mar.hjelm@gmail.com

 % If not set, set default values for the number of partitions, repititions, 
  % and the loss function.
  if isempty(N_folds)
    N_folds = 10;
  end
  
  if isempty(N_repetitions)
   N_repetitions = 5;
  end
  
  % Check erroneous input
  if nargin < 4
      error('cv.m: Too few input arguments. For help type: help cv.m\n');    
  end

  
  % Get number of datapoints
  [N_samples,~] = size ( X );   
  
 
  if N_folds == 0
    % This means leave one out cross-validation
    N_folds = N_samples;
  end
  

  if ~numel(varargin)
   loss_function = @zero_one_loss;
  else 
   loss_function = varargin{1};
  end
  
  print = 0;
  

  
  % Extract number of parameter types:  N_param/2
  [~,N_param_types] = size ( params ); 
  N_param_types = N_param_types / 2;
 
  % We need a set containing every parameter combination such that we can 
  % iterate through all the possible combinations. The combo function takes the
  % cardinality of every parameter and returns index sets for all possible
  % combinations of parameters

  % Treat this as a search over a grid. 
  paramset = cell(1,N_param_types);
  c = cell(1,N_param_types);
  counter = 1;
 
  for i = 2:2:2*N_param_types
    paramset{counter} = 1:size( params { 1, i }, 2 );
    counter=1+counter;
  end
  [c{:}] = ndgrid( paramset{:} );
  parameter_combinations = cell2mat( cellfun(@(v)v(:), c, 'UniformOutput',false) );

  % Create variable to hold errors for the different combinations
  [N_combinations, ~] = size ( parameter_combinations );
  average_error_rate = zeros ( N_combinations, 1 );


  dps = floor ( N_samples / N_folds ); % Data Per Set
  mod_partition = mod ( N_samples, N_folds );  % Data that is left out 

  % Disperse uneven data points by adding one to each set until we run
  % out...
  pPos = zeros(1,N_folds+1);
  pPos(1) = 1;
  for i = 2:N_folds+1
    if mod_partition == 0
      pPos(i) = pPos(i-1) + dps;
    else 
      pPos(i) = pPos(i-1) + dps + 1;
      mod_partition = mod_partition-1;
    end
  end
  
  % Check that this works

 
  % ***** START CROSSVALIDATION RUN *****
  if print
    disp(' ') 
    disp ( 'Beam me up Scotty' );
    disp ( ['Starting cross-validation with ',num2str(N_repetitions),' repititions and ',num2str(N_folds), ' partitions of the data. '] );
    disp ( 'This will take a while so go grab a coffee and a cinnamon bun...' )
  end
  total_time = 0;  
  total_time_sum = 0;
  
  % Start  cross-validation algorithm and do it for nrepetitions
  for repitition = 1:N_repetitions
    
    tStart = tic;
    if print
      disp(' ');
      disp ( ['Starting repitition ',num2str(repitition), ' of ', num2str(N_repetitions) ] );
    end

    % PARTITION DATA
    % Partition data into ('nfolds' = k) random sets
    % We have k sets and n vectors thus n/k vectors in each set, so all we
    % have to do is create a randomly permutation vector of the number of  
    % data points  i.e. 1 2 3 4...N.  The i:th set then consist of the
    % i*k + n/k indices in the permutation vector which we plug into the big
    % data matrix to get the points.
    rng('shuffle');
    permIdxs = randperm ( N_samples );    
    
    
    
    % Wikipedia
    % In k-fold cross-validation, the original sample is randomly 
    % partitioned into k equal size subsamples. Of the k subsamples, a 
    % single subsample is retained as the validation data for testing the 
    % model, and the remaining k ? 1 subsamples are used as training data. 
    % The cross-validation process is then repeated k times (the folds), 
    % with each of the k subsamples used exactly once as the validation 
    % data. The k results from the folds can then be averaged (or otherwise
    % combined) to produce a single estimation.
    
    
    
    % For each partition do
    for j_partition = 1:N_folds      
     
      tmpIdx = pPos(j_partition):pPos(j_partition+1)-1;   
      tmp2Idx = 1:N_samples;
      tmp2Idx(tmpIdx) = [];

      trIdx = permIdxs( tmp2Idx );
      teIdx = permIdxs( tmpIdx );


         
      % ### DO TRAINING AND PREDICTION FOR EVERY PARAMETER COMBINATION ###
      
      % For all parameter combinations do
      for p = 1:N_combinations
        
        % Create empty cell to hold parameter values
        varargin = cell(1,N_param_types);
        
        % Create cell containing parameter labels and values
        for pt = 1:N_param_types
       
          param_values =  params { 1, 2*(pt-1)+2 };
          varargin{ 1, pt } = param_values ( 1, parameter_combinations ( p, pt ) );        
          
        end
        
        % Train classifier on training data. C is an object holding the
        % classifier trained with the params and a function handle for
        % predicton
        C = feval ( classifier_handle, X(trIdx,:), y(trIdx,:), varargin{:} ) ;

        % Do prediction on test data
        y_pred = feval ( C.applyfunc, C, X(teIdx,:) );

        % Sum up and average to get the average error estimate
        average_error_rate(p) = average_error_rate(p) + feval( loss_function, y(teIdx,:), y_pred );
      
      end
      
      % ######################################################

      
    end
    % End for
    
    total_time = toc ( tStart );
    total_time_sum = total_time_sum + total_time;    
    
    if print 
      disp('-----------------------------------------');  
      disp(['Repition error: ',sprintf('%10.2f',100.*average_error_rate./(N_folds*repitition))]);
      %disp(['Repition total average error: ',sprintf('%1.2f',mean(100.*average_error_rate./(N_folds*repitition)))]);
      disp ( ['Total time and counting: ', num2str(total_time), 's']);     
      disp('-----------------------------------------')  
    end
    
  end
  % End algorithm
  
  if print 
    disp ( ['Ended with ', num2str(N_folds*N_repetitions),' trials,',...
    ' total running time of ',num2str(total_time_sum), ' seconds and',sprintf('\n'),...
    ' an average time of ', num2str(total_time_sum/(N_folds*N_repetitions)),...
    ' seconds per training set.'])
  end
  % Set final average error rate
  average_error_rate = ( 1 / ( N_folds * N_repetitions ) )  .* average_error_rate;
  
  
  % SET OPTIMAL PARAMETER COMBINATION 
  
  % Find the min average error rate and get the parameter values for that combination 
  [~, I] = min ( average_error_rate );
  best_parameter_combination = parameter_combinations( I , :);
  
  % Create empty cell to hold optimal parameter values
  opt_param = cell ( 1, 2*N_param_types );
  varargin = cell ( 1, N_param_types );
  
  % Put in all parameter labels and values
  for pt = 1:2:(2*N_param_types-1)
  
    % Assign parameter label
    opt_param { 1, pt } =   params { 1, pt };
    
    % Assign parameter value
    param_values = params { 1, pt+1 };    
    opt_param{ 1, pt+1 } = param_values( 1, best_parameter_combination ( 1, (pt+1)/2 ) );
    varargin{ 1, (pt+1)/2  } = param_values( 1, best_parameter_combination ( 1, (pt+1)/2 ) );        
    
  end  
  
  
  % Calculate the values with optimal parameters  
  C = feval ( classifier_handle, X, y, varargin{:} ) ;
  
  if print 
    disp('Optimal params:')
    disp( opt_param )
    disp(' ')
  end
    
end
% End function



function l = zero_one_loss ( y_true, y_pred )
  % Calculates the average 0/1-loss ratio
  l = sum ( ( y_pred ~= y_true ) , 1 ) / size ( y_pred, 1 );
end
