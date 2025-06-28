
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

function DestinationEagle = DetermineDestinationEagles (ArchiveF, nobj, options)

% find limiting agents and calculate crowding scores
[CrowdingScores, ~] = CalcCrowdingScores (ArchiveF);

% sample from the front weighted by crowding scores 
[~, DestinationEagle] = datasample(CrowdingScores, options.PopulationSize, 1, 'Weights', CrowdingScores);

DestinationEagle = DestinationEagle';

end
