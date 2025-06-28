
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

function [fun,nobj,nvars,lb,ub] = ...
	GetFunctionDetails (FunctionNumber)

switch FunctionNumber
	case 1
		fun   = f_cec09 (sprintf('UF%d',FunctionNumber));
		nobj  = 2;
		nvars = 30;
	case 2
		fun   = f_cec09 (sprintf('UF%d',FunctionNumber));
		nobj  = 2;
		nvars = 30;
	case 3
		fun   = f_cec09 (sprintf('UF%d',FunctionNumber));
		nobj  = 2;
		nvars = 30;
	case 4
		fun   = f_cec09 (sprintf('UF%d',FunctionNumber));
		nobj  = 2;
		nvars = 30;
	case 5
		fun   = f_cec09 (sprintf('UF%d',FunctionNumber));
		nobj  = 2;
		nvars = 30;
	case 6
		fun   = f_cec09 (sprintf('UF%d',FunctionNumber));
		nobj  = 2;
		nvars = 30;
	case 7
		fun   = f_cec09 (sprintf('UF%d',FunctionNumber));
		nobj  = 2;
		nvars = 30;
	case 8
		fun   = f_cec09 (sprintf('UF%d',FunctionNumber));
		nobj  = 3;
		nvars = 30;
	case 9
		fun   = f_cec09 (sprintf('UF%d',FunctionNumber));
		nobj  = 3;
		nvars = 30;
	case 10
		fun   = f_cec09 (sprintf('UF%d',FunctionNumber));
		nobj  = 3;
		nvars = 10;
	otherwise
		fun   = f_cec09 (sprintf('UF%d',1));
		nobj  = 2;
		nvars = 30;
end

xrange  = xboundary (sprintf('UF%d',FunctionNumber),nvars);
lb      = xrange (:, 1)';
ub      = xrange (:, 2)';
