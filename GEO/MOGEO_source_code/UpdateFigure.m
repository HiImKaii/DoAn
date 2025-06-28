
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

function UpdateFigure (FitnessScores,ArchiveF,CurrentIteration,nobj,nvars,options)

persistent FgrHdl_FitnessScores FgrHdl_ArchiveF

if and (CurrentIteration>1, CurrentIteration<options.MaxIterations)
	
	switch nobj
		
		case 2
			
			FgrHdl_FitnessScores.XData = FitnessScores(:,1);
			FgrHdl_FitnessScores.YData = FitnessScores(:,2);
			FgrHdl_ArchiveF.XData = ArchiveF(:,1);
			FgrHdl_ArchiveF.YData = ArchiveF(:,2);
			fprintf ('Current iteration: %d / %d\t\t# Archive members: %d / %d\n',CurrentIteration,options.MaxIterations,size(ArchiveF,1),options.ArchiveSize)
			
		case 3
			
			FgrHdl_FitnessScores.XData = FitnessScores(:,1);
			FgrHdl_FitnessScores.YData = FitnessScores(:,2);
			FgrHdl_FitnessScores.ZData = FitnessScores(:,3);
			FgrHdl_ArchiveF.XData = ArchiveF(:,1);
			FgrHdl_ArchiveF.YData = ArchiveF(:,2);
			FgrHdl_ArchiveF.ZData = ArchiveF(:,3);
			fprintf ('Current iteration: %d / %d\t\t# Archive members: %d / %d\n',CurrentIteration,options.MaxIterations,size(ArchiveF,1),options.ArchiveSize)
			
	end
	
elseif CurrentIteration == 1
	
	close 'all';
	figure ('Position',[756,162,560,420]);
	[FrontF_optimal, ~] = pareto (sprintf('UF%d',options.FunctionNumber), 500, nvars);
	TrueParetoF = FrontF_optimal';
	
	switch nobj
		
		case 2
			hold on
			scatter (TrueParetoF(:,1), TrueParetoF(:,2), ...
				6, 'b', 'filled', ...
				'DisplayName', 'True Pareto Front');
			FgrHdl_FitnessScores = scatter (FitnessScores(:,1),FitnessScores(:,2), ...
				6, 'k', 'filled', ...
				'DisplayName', 'Golden eagles');
			FgrHdl_ArchiveF = scatter (ArchiveF(:,1),ArchiveF(:,2), ...
				36, ...
				'MarkerFaceColor', 'r', ...
				'MarkerEdgeColor', 'k', ...
				'DisplayName', 'Current archive');
			hold off
			
		case 3
			
			hold on
			scatter3 (TrueParetoF(:,1), TrueParetoF(:,2), TrueParetoF(:,3),...
				6, 'b', 'filled', ...
				'DisplayName', 'True Pareto Front');
			FgrHdl_FitnessScores = scatter3 (FitnessScores(:,1), FitnessScores(:,2), FitnessScores(:,3), ...
				6, 'k', 'filled', ...
				'DisplayName', 'Golden eagles');
			FgrHdl_ArchiveF = scatter3 (ArchiveF(:,1), ArchiveF(:,2), ArchiveF(:,3), ...
				36, ...
				'MarkerFaceColor', 'r', ...
				'MarkerEdgeColor', 'k', ...
				'DisplayName', 'Current archive');
			hold off
			view ([142.5,10]);
			
	end
	
	box ('on');
	grid ('on');
	legend ();
	fprintf ('Current iteration: %d / %d\t\t# Archive members: %d / %d\n',CurrentIteration,options.MaxIterations,size(ArchiveF,1),options.ArchiveSize);

else
	
	delete (FgrHdl_FitnessScores);
	FgrHdl_ArchiveF.DisplayName = 'Obtained Pareto front';
	fprintf ('Current iteration: %d / %d\t\t# Archive members: %d / %d\n',CurrentIteration,options.MaxIterations,size(ArchiveF,1),options.ArchiveSize);
	
end

drawnow limitrate;

end


