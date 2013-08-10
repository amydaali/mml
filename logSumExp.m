function lse = logSumExp ( X )
% function lse = logSumExp ( X )
%
% usage
% lse = logSumExp ( X )
%
% input
% 	X : (M x N)-matrix of column vectors of log values.
%
% output
% 	lse : The log sum exponential over all columns.
%
% description
% 	Computes the log sum exponential for a set of small values. It is used
% 	to avoid numerical underflow when calculcating the log of a sum. 
%
% 	Each column value of X is a log value from a mixture or some other that  
%	  we want to sum. If the matrix contain a different number of components the unfilled 
%	  values need to be set to 0. For example: 
%	  X = [log(pi11) + log(P11(x1)), log(pi12) + log(P11(x2));
%		     log(pi21) + log(P21(x1)), log(pi22) + log(P21(x2));
%		     log(pi31) + log(P31(x1)), 0]
%	  The 0 number will be disregarded in the computations.
%
%	  A brief description of how this works: 
% 	In a mixture model the probability of an event x is 
% 	P(x) = pi1*P1(x) + pi2*P2(x)... 
% 	The problem is usually that  P1, P2,... are small for all x which makes 
% 	underflow happen. To fix underflow one usually operates in the log 
% 	domain i.e. log(P(x)) = log(pi1*P1(x) + pi2*P2(x)...)
% 	The problem with this is that the log cannot decompose sums and we 
% 	still get underflow. To fix this(somewhat) we can write:
% 	log(P(x)) = log( exp(log(pi1) + log(P1(x)) ) + 
%		log( exp(log(pi2) + log(P2(x)) ) + ...). Now by finding the max value
%		of pi1*P1(x),pi2*P2(x),... and deducting it we can remove most of the
%		value in the equation and get it out. It is simple if one looks at the 
%		following calculations
%
%		log(p) = log(p1 + p2) = log(exp(log(p1))+exp(log(p2)))
%		pMax = max([log(p1),log(p2)])
%		log(p) = log( exp(pMax) * ( exp(log(p1)-pMax)+exp(log(p2-pMax)) )
%		log(p) = pMax + log( exp(log(p1)-pMax) + exp(log(p2)-pMax) )
% 	
%		Now if we for example assume log(p1)>log(p2) then 
%		log(p) = pMax + log( exp(0) + exp(log(p2)-pMax) ) =
%		pMax + log( 1 + exp(log(p2)-pMax) )
%		
% 	This means that we gotten out most of the probability mass from the 
%		sum and we have avoided summing several small numbers. Hopefully 
%		the exp(log(p2)-pMax) will be nice as well. 
% 	
% author
%     Martin Hjelm, mar.hjelm@gmail.com
%
% copyright
%     Do what ever you want but give me credit, if credit is due.
%


%%%%%%%%%% CHECK INPUT ETC. %%%%%%%%%%%%

% Check erroneous input
  if nargin < 1
      error('PCA.m: Too few input arguments. For help type help PCA.\n');    
  end  
  
 	[M,N] = size(X);
 	% Get max in all columns
 	xMax = max ( X );
 	% Subtract max in all columns from all column values and 
 	% calculate the final log sum exp
 	lse = zeros(1,N);
 	for i_mixtures = 1:N
 		mixComps = X(:,i_mixtures);
 		mixComps(mixComps==0) = [];
 		lse(1,i_mixtures) = xMax(i_mixtures) + log ( sum ( exp( mixComps - xMax(i_mixtures) ) ) );
 	end

 	% Check for infinity and if so take xMax as the value
  lseInf = ~isfinite(lse);
  lse(lseInf) = xMax(lseInf);  

end
