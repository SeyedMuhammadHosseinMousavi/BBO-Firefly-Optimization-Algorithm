function bestfis=FireFlyFCN(fis,data)
% Variables
p0=GettingFuzzyParameters(fis);
Problem.CostFunction=@(x) FuzzyCost(x,fis,data);
Problem.nVar=numel(p0);
alpha=1;
Problem.VarMin=-(10^alpha);
Problem.VarMax=10^alpha;
%
% FireFly Parameters
Params.MaxIt=10;
Params.nPop=6;
%
% Starting FireFly Algorithm
results=Runfly(Problem,Params);
%
% Getting the Results
p=results.BestSol.Position.*p0;
bestfis=FuzzyParameters(fis,p);
end
%%
function results=Runfly(Problem,Params)
% disp('Starting FireFly Training');
%
% Cost Function
CostFunction=Problem.CostFunction;  
% Number of Decision Variables
nVar=Problem.nVar;   
% Size of Decision Variables Matrixv
VarSize=[1 nVar]; 
% Lower Bound of Variables
VarMin=Problem.VarMin;    
% Upper Bound of Variables
VarMax=Problem.VarMax;      
% Some Change
if isscalar(VarMin) && isscalar(VarMax)
dmax = (VarMax-VarMin)*sqrt(nVar);
else
dmax = norm(VarMax-VarMin);
end
%
%% FireFly Algorithm Parameters
% Maximum Number of Iterations
MaxIt=Params.MaxIt;
% Number of Fireflies (Swarm Size)
nPop=Params.nPop;    
% Light Absorption Coefficient
gamma = 1;   
% Attraction Coefficient Base Value
beta0 = 2;    
% Mutation Coefficient
alpha = 0.2;     
% Mutation Coefficient Damping Ratio
alpha_damp = 0.98;    
% Uniform Mutation Range
delta = 0.05*(VarMax-VarMin);     
m = 2;
%
% Second Stage
% Empty Firefly Structure
empty_firefly.Position = [];
empty_firefly.Cost = [];
% Initialize Population Array
pop = repmat(empty_firefly, nPop, 1);
% Initialize Best Solution Ever Found
BestSol.Cost = inf;
% Create Initial Fireflies
for i = 1:nPop
pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
pop(i).Cost = CostFunction(pop(i).Position);
if pop(i).Cost <= BestSol.Cost
BestSol = pop(i);
end
end
% Array to Hold Best Cost Values
BestCost = zeros(MaxIt, 1);
%
%% Firefly Algorithm Main Body
%
for it = 1:MaxIt
newpop = repmat(empty_firefly, nPop, 1);
for i = 1:nPop
newpop(i).Cost = inf;
for j = 1:nPop
if pop(j).Cost < pop(i).Cost
rij = norm(pop(i).Position-pop(j).Position)/dmax;
beta = beta0*exp(-gamma*rij^m);
e = delta*unifrnd(-1, +1, VarSize);
newsol.Position = pop(i).Position ...
+ beta*rand(VarSize).*(pop(j).Position-pop(i).Position) ...
+ alpha*e;
newsol.Position = max(newsol.Position, VarMin);
newsol.Position = min(newsol.Position, VarMax);
newsol.Cost = CostFunction(newsol.Position);
if newsol.Cost <= newpop(i).Cost
newpop(i) = newsol;
if newpop(i).Cost <= BestSol.Cost
BestSol = newpop(i);
end
end
end
end
end
% Merge Operation
pop = [pop
newpop];
% Sort Operation
[~, SortOrder] = sort([pop.Cost]);
pop = pop(SortOrder);
% Truncate Operation
pop = pop(1:nPop);
% Store Best Cost Ever Found
BestCost(it) = BestSol.Cost;
% Show Iteration Information
disp(['In Iteration ' num2str(it) ' : BBO-FireFly Fittest Cost Is =  ' num2str(BestCost(it))]);
% Damp Mutation Coefficient
alpha = alpha*alpha_damp;
end
disp('BBO-FireFly Algorithm Came To End');
% Store Res
results.BestSol=BestSol;
results.BestCost=BestCost;
% Plot FireFly Training Stages
figure;
set(gcf, 'Position',  [600, 300, 500, 300])
plot(BestCost,'-',...
'LineWidth',2,...
'MarkerSize',3,...
'MarkerEdgeColor','k',...
'Color',[0.1,0.1,0.1]);
title('FireFly Algorithm Training','FontSize',10,...
'FontWeight','bold','Color','b');
xlabel('FireFly Iteration Number','FontSize',10,...
'FontWeight','bold','Color','k');
ylabel('FireFly Best Cost Result','FontSize',10,...
'FontWeight','bold','Color','k');
legend({'FireFly Algorithm Train'});
end


