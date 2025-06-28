
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
%		  the .m file you define should accept 'x' as input and return 
%		  a column vector containing objective function values 
% nobj  : number of objectives 
% nvars : number of decision/design variables 
% lb    : lower bound of decision variables (must be of size 1 x nvars)
% ub    : upper bound of decision variables (must be of size 1 x nvars)
%
% MOGEO will return the following: 
% x     : best solution found 
% fval  : objective function values of the found Pareto solution 

%% Inputs 

FunctionNumber = 10; % 1-10

options.PopulationSize = 200;
options.ArchiveSize    = 100;
options.MaxIterations  = 1000;

options.FunctionNumber = FunctionNumber;

%% Run Multi-Objective Golden Eagle Optimizer 

[fun,nobj,nvars,lb,ub]   = GetFunctionDetails (FunctionNumber);

options.AttackPropensity = [0.5 ,   2];
options.CruisePropensity = [1   , 0.5];

[x,fval] = MOGEO (fun,nobj,nvars,lb,ub,options);
