
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

function [FrontX, FrontF] = UpdateArchive (x, nobj, FitnessScores, FrontX, FrontF, stage, options)

switch stage
	
	case 1
		
		% -------------------------------------
		%       stage 1 : initialize front
		% -------------------------------------
		
		% find non-dominated solutions for each objective
		Domination = zeros(options.PopulationSize, options.PopulationSize, nobj);
		for i1 = 1:options.PopulationSize
			Domination(:,i1,:) = permute(FitnessScores < FitnessScores(i1,:), [1,3,2]);
		end

		% find dominated solutions' index
		[~,DominatedIndex] = find(sum(Domination,3)==nobj);
		
		% find unique indexes in DominatedIndex
		DominatedIndexUnique = unique(DominatedIndex);
		NonDominatedIndex = setdiff((1:options.PopulationSize)', DominatedIndexUnique);
		
		% construct the front
		if numel(NonDominatedIndex) <= options.ArchiveSize % if there are few non-dominated agents, transfer all of them to the front
			
			% construct front's x and fval
			FrontX = x(NonDominatedIndex,:);
			FrontF = FitnessScores(NonDominatedIndex,:);
			
		else % if there are too many non-dominated agents, sample from them weighted by crowding score. Limiting solutions are invariably selected for front
			
			% find limiting agents and calculate crowding scores
			[CrowdingScores, LimitingAgents] = CalcCrowdingScores (FitnessScores(NonDominatedIndex,:));
			
			% initialize the mask for selected agents to be transferred to front
			SelectedAgents = false(options.PopulationSize, 1);
			
			% limiting agents are selected directly
% 			SelectedAgents(LimitingAgents) = true;
			
			% prepare crowding scores for sampling. larger crowding scores are better
% 			CrowdingScores = 1 - CrowdingScores;
% 			CrowdingScores(LimitingAgents) = eps;
			
			% sample from the non-dominated solutions 
			idx = datasample(NonDominatedIndex, options.ArchiveSize, 1, 'Weights', CrowdingScores, 'Replace', false);
			
			% set the mask
			SelectedAgents(idx) = true;
			
			% transfer non-dominated solutions to front 
			FrontX = x(SelectedAgents,:);
			FrontF = FitnessScores(SelectedAgents,:);
			
		end
		
	case 2
		
		% -------------------------------------
		%       stage 2 : update front
		% -------------------------------------
		
		for i1 = 1:options.PopulationSize
			
			% test which members of the front are dominated by this agent
			FitnessComparison = sum(FitnessScores(i1,:) < FrontF, 2);
			
			if any (FitnessComparison == nobj) % the agent dominates one or more of the front's members
				
				% create a mask for dominated members of the front 
				DominatedMask = FitnessComparison == nobj;
				
				% remove dominated front members
				FrontX(DominatedMask,:) = [];
				FrontF(DominatedMask,:) = [];
				
				% add the new member
				FrontX = vertcat(FrontX, x(i1,:)); %#ok
				FrontF = vertcat(FrontF, FitnessScores(i1,:)); %#ok
				
			elseif all (FitnessComparison>0 & FitnessComparison<nobj) % the agent is nondominated and should be inserted into the front
				
				if size(FrontF, 1) < options.ArchiveSize % front is not full, simply add the agent to the front
					FrontX = vertcat(FrontX, x(i1,:)); %#ok
					FrontF = vertcat(FrontF, FitnessScores(i1,:)); %#ok
					
				else % front is full, randomly select a front member that is located in dense areas of the front based on crowding score
					
					% calculate crowding scores
					[CrowdingScores, LimitingAgents] = CalcCrowdingScores (FrontF);
					CrowdingScores = 1 - CrowdingScores;
% 					CrowdingScores(LimitingAgents) = eps;
					
					% select one of the front members (limiting agents cannot be selected)
					[~, idx] = datasample(CrowdingScores, 1, 1, 'Weights', CrowdingScores, 'Replace', false);
					
					% replace the exiting member with the new member
					FrontX(idx,:) = x(i1,:);
					FrontF(idx,:) = FitnessScores(i1,:);
					
				end
				
			end
			
		end
		
end






