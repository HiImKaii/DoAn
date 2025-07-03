%% White Shark Optimizer (WSO) source codes version 1.0
%
%  Developed in MATLAB R2018a
%
%  Programmer: Malik Braik
%
%         e-Mail: mbraik@bau.edu.jo
%

%   Main paper:
%   Malik Braik, Abdelaziz Hammouri, Jaffar Atwan, Mohammed Azmi Al-Betar, Mohammed A.Awadallah

%   White Shark Optimizer: A novel bio-inspired meta-heuristic algorithm for global optimization problems
%   Knowledge-Based Systems
%   DOI: https://doi.org/10.1016/j.knosys.2022.108457
%____________________________________________________________________________________
%%   
clear 
close all
clc
%% % Prepare the problem
dim = 2;
ub = 50 * ones(1, 2);
lb = -50 * ones(1, 2);
fobj = @Objfun;

%% % CSA parameters 
searchAgents = 30;
maxIter = 1000;
  
              [fitness,gbest,ccurve]=WSO(searchAgents,maxIter,lb,ub,dim,fobj);
                     
              disp(['===> The optimal fitness value found by Standard Chameleon is ', num2str(fitness, 12)]);

%% Draw the convergence behavior curve
         
figure;  set(gcf,'color','w');

plot(ccurve,'LineWidth',2,'Color','b'); grid;
title({'Convergence characteristic curve'},'interpreter','latex','FontName','Times','fontsize',10);
xlabel('Iteration','interpreter','latex','FontName','Times','fontsize',10)
ylabel('Best score obtained so far','interpreter','latex','FontName','Times','fontsize',10); 

axis tight; grid on; box on 
     
h1=legend('WSO','location','northeast');
set(h1,'interpreter','Latex','FontName','Times','FontSize',10) 
ah=axes('position',get(gca,'position'),...
            'visible','off');