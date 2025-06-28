
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

% To use this code in your own project 
% remove the line for 'GetFunctionDetails' function 
% and define the following parameters: 
% fun   : function handle to the .m file containing the objective function
%		  the .m file you define should accept the whole population 'x' 
%		  as input and return a column vector containing objective function 
%		  values of all of the population members
% nvars : number of decision/design variables 
% lb    : lower bound of decision variables (must be of size 1 x nvars)
% ub    : upper bound of decision variables (must be of size 1 x nvars)
%
% MOGEO will return the following: 
% x     : best Pareto-optimal solution found 
% fval  : objective function values of the Pareto Pareto-optimal found

function [x,fval] = MOGEO (fun,nobj,nvars,lb,ub,options)

%% initialization

stage = 1;

PopulationSize = options.PopulationSize;
MaxIterations = options.MaxIterations;

x = lb + rand (PopulationSize,nvars) .* (ub-lb);

FitnessScores = fun (x);

AttackPropensity = linspace (options.AttackPropensity(1), options.AttackPropensity(2), MaxIterations);
CruisePropensity = linspace (options.CruisePropensity(1), options.CruisePropensity(2), MaxIterations);

% initialize front
[ArchiveX, ArchiveF] = UpdateArchive (x, nobj, FitnessScores, [], [], stage, options);

%% main loop

stage = 2; % iterations stage

for CurrentIteration = 1 : MaxIterations
	
	% modify archive size to have at least 3 members
	if size(ArchiveF, 1) < 3
		NumSample = 3 - size(ArchiveF, 1);
		[~, SampleIdx] = datasample (ArchiveF, NumSample, 1);
		ArchiveF = vertcat (ArchiveF, ArchiveF(SampleIdx,:)); %#ok
		ArchiveX = vertcat (ArchiveX, ArchiveX(SampleIdx,:)); %#ok
	end
	
	% choose destination eagles
	DestinationEagle = DetermineDestinationEagles (ArchiveF, nobj, options);
	
	% calculate AttackVectorInitial
	AttackVectorInitial = ArchiveX (DestinationEagle,:) - x;
	
	% calculate Radius
	Radius = VecNorm (AttackVectorInitial,2,2);
	
	% determine converged and unconverged eagles
	ConvergedEagles = sum (Radius,2) == 0;
	UnconvergedEagles = ~ ConvergedEagles;
	
	% initialize CruiseVectorInitial
	CruiseVectorInitial = 2 .* rand(PopulationSize, nvars) - 1; % [-1,1]
	
	% correct vectors for converged eagles
	AttackVectorInitial (ConvergedEagles, :) = 0;
	CruiseVectorInitial (ConvergedEagles, :) = 0;
	
	% determine constrained and free variables
	for i1 = 1 : PopulationSize
		if UnconvergedEagles (i1)
			vConstrained = false ([1, nvars]); % mask
			idx = datasample (find(AttackVectorInitial(i1,:)), 1, 2);
			vConstrained (idx) = 1;
			vFree = ~vConstrained;
			CruiseVectorInitial (i1,idx) = - sum(AttackVectorInitial(i1,vFree).*CruiseVectorInitial(i1,vFree), 2) ./ (AttackVectorInitial(i1,vConstrained));
		end
	end
	
	% calculate unit vectors
	AttackVectorUnit = AttackVectorInitial ./ VecNorm (AttackVectorInitial, 2, 2);
	CruiseVectorUnit = CruiseVectorInitial ./ VecNorm (CruiseVectorInitial, 2, 2);
	
	% correct vectors for converged eagles
	AttackVectorUnit (ConvergedEagles, :) = 0;
	CruiseVectorUnit (ConvergedEagles, :) = 0;
	
	% calculate movement vectors
	AttackVector = rand (PopulationSize, 1) .* AttackPropensity (CurrentIteration) .* Radius .* AttackVectorUnit;
	CruiseVector = rand (PopulationSize, 1) .* CruisePropensity (CurrentIteration) .* Radius .* CruiseVectorUnit;
	StepVector = AttackVector + CruiseVector;
	
	% calculate new x
	x = x + StepVector;
	
	% enforce bounds
	lbExtended = repmat (lb, [PopulationSize, 1]);
	ubExtended = repmat (ub, [PopulationSize, 1]);
	
	lbViolated = x < lbExtended;
	ubViolated = x > ubExtended;
	
	x (lbViolated) = lbExtended (lbViolated);
	x (ubViolated) = ubExtended (ubViolated);
	
	% calculate fitness function
	FitnessScores = fun (x);
	
	% update archive
	[ArchiveX, ArchiveF] = UpdateArchive (x,nobj,FitnessScores,ArchiveX,ArchiveF,stage,options);
	
	% update plot
	UpdateFigure (FitnessScores,ArchiveF,CurrentIteration,nobj,nvars,options);
	
end

%% return values

x = ArchiveX;
fval = ArchiveF;

