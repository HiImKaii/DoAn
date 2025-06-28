
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  
%  Multi-Objective Golden Eagle Optimizer (MOGEO) source codes version 1.0
%  
%  Developed in:	MATLAB 9.6 (R2019a)
%  
%  Programmer:		Abdolkarim Mohammadi-Balani
%  
%  Original paper:	Abdolkarim Mohammadi-Balani, Mahmoud Dehghan Nayeri, 
%					Adel Azar, Mohammadreza Taghizadeh-Yazdi, 
%					Golden Eagle Optimizer: A nature-inspired 
%					metaheuristic algorithm, Computers & Industrial Engineering.
%
%                  https://doi.org/10.1016/j.cie.2020.107050               
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [CrowdingScores, LimitingAgents] = CalcCrowdingScores (FrontF)

% find limiting agents ----------------------------------------

[~, MinIndex] = mink(FrontF, 1);
[~, MaxIndex] = maxk(FrontF, 1);
LimitingAgents = unique(vertcat(MinIndex', MaxIndex'));

% calculate crowding scores -----------------------------------

% normalize FrontF
FrontFNormalized = (FrontF-min(FrontF)) ./ (max(FrontF)-min(FrontF));

% sort F
[SortedFrontFNormalized, Index] = sort(FrontFNormalized);

% calculate initial crowding scores for middle pack
SortedCrowdingScores = zeros(size(FrontF));
SortedCrowdingScores(2:end-1,:) = SortedFrontFNormalized(3:end,:)-SortedFrontFNormalized(1:end-2,:);

% calculate initial crowding scores for limiting agents
SortedCrowdingScores(1,:) = SortedFrontFNormalized(2,:);
SortedCrowdingScores(end,:) = SortedFrontFNormalized(end,:)-SortedFrontFNormalized(end-1,:);

% place scores in their original place
UnsortedCrowdingScores = zeros(size(FrontF));
for i1 = 1:size(FrontF, 2)
	UnsortedCrowdingScores(Index(:,i1),i1) = SortedCrowdingScores(:,i1);
end

% calculate crowding scores
CrowdingScores = mean(UnsortedCrowdingScores, 2); % ensure crowding scores are always betwen 0 and 1

% handle NaN or zero weights ----------------------------------------------------

CrowdingScores(CrowdingScores==0) = eps;
CrowdingScores(isnan(CrowdingScores)) = eps;
