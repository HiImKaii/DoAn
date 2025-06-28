
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

function PlotResults (fval)

% ----------------

f1 = linspace(0,1,500)';
g  = 1;
h  = 1 - sqrt(f1./g);
f2 = g .* h;
TrueParetoF = horzcat(f1,f2);

% ----------------

figure ('Position',[795,199,440,395]);

hold on
scatter (TrueParetoF(:,1),TrueParetoF(:,2), 9, 'b', 'filled', ...
	'DisplayName','True Pareto front');
scatter (fval(:,1),fval(:,2), ...);
	'MarkerFaceColor', 'r', ...
	'MarkerEdgeColor', 'k', ...
	'DisplayName', 'Obtained Pareto front');
hold off

daspect ([1,1,1]);

xlim ([0,1.2]);
ylim ([0,1.2]);

xlabel ('f_1');
ylabel ('f_2');

box ('on');
grid ('on');

legend ();




