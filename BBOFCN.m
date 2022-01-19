function bestfis=BBOFCN(fis,data)
% Variables
p0=GettingFuzzyParameters(fis);
Problem.CostFunction=@(x) FuzzyCost(x,fis,data);
Problem.nVar=numel(p0);
alpha=1;
VarMin = -10;         % Decision Variables Lower Bound
VarMax = 10;          % Decision Variables Upper Bound
Problem.VarMin=-(10^alpha);
Problem.VarMax=10^alpha;
%
% BBO Algorithm Parameters
Params.MaxIt = 10;          % Maximum Number of Iterations
Params.nPop = 7;            % Number of Habitats (Population Size)
Params.KeepRate = 0.4;      % Keep Rate
Params.nKeep = round(Params.KeepRate*Params.nPop);     % Number of Kept Habitats
Params.nNew = Params.nPop-Params.nKeep;                % Number of New Habitats

% Migration Rates
Params.mu = linspace(1, 0, Params.nPop);        % Emmigration Rates
Params.lambda = 1-Params.mu;                    % Immigration Rates
Params.alpha = 0.9;
Params.pMutation = 0.1;
Params.sigma = 0.02*(VarMax-VarMin);
%
% Starting BBO Algorithm
results=RunbboFCN(Problem,Params);
% Getting the Results
p=results.BestSol.Position.*p0;
bestfis=FuzzyParameters(fis,p);
end
%%----------------------------------------------
function results=RunbboFCN(Problem,Params)
disp('Starting BBO-FireFly Algorithm Training');
%------------------------------------------------
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
%--------------------------------------------
%% BBO Algorithm Parameters

MaxIt = Params.MaxIt;          % Maximum Number of Iterations
nPop = Params.nPop;            % Number of Habitats (Population Size)
KeepRate =Params.KeepRate;     % Keep Rate
nKeep = Params.nKeep;          % Number of Kept Habitats
nNew = Params.nNew;            % Number of New Habitats
% Migration Rates
mu = Params.mu;                 % Emmigration Rates
lambda = Params.lambda;         % Immigration Rates
alpha = Params.alpha;
pMutation = Params.pMutation;
sigma = Params.sigma;
%------------------------------------------------------
%% Second Stage
% Empty Habitat
habitat.Position = [];
habitat.Cost = [];
% Create Habitats Array
pop = repmat(habitat, nPop, 1);
% Initialize Habitats
for i = 1:nPop
pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
pop(i).Cost = CostFunction(pop(i).Position);
end
% Sort Population
[~, SortOrder] = sort([pop.Cost]);
pop = pop(SortOrder);
% Best Solution Ever Found
BestSol = pop(1);
% Array to Hold Best Costs
BestCost = zeros(MaxIt, 1);
%
%% BBO Algorithm Main Body
for it = 1:MaxIt
newpop = pop;
for i = 1:nPop
for k = 1:nVar
% Migration
if rand <= lambda(i)
% Emmigration Probabilities
EP = mu;
EP(i) = 0;
EP = EP/sum(EP);
% Select Source Habitat
j = RouletteWheelSelection(EP);
% Migration
newpop(i).Position(k) = pop(i).Position(k) ...
+alpha*(pop(j).Position(k)-pop(i).Position(k));
end
% Mutation
if rand <= pMutation
newpop(i).Position(k) = newpop(i).Position(k)+sigma*randn;
end
end
% Apply Lower and Upper Bound Limits
newpop(i).Position = max(newpop(i).Position, VarMin);
newpop(i).Position = min(newpop(i).Position, VarMax);
% Evaluation
newpop(i).Cost = CostFunction(newpop(i).Position);
end
% Sort New Population
[~, SortOrder] = sort([newpop.Cost]);
newpop = newpop(SortOrder);
% Select Next Iteration Population
pop = [pop(1:nKeep)
newpop(1:nNew)];
% Sort Population
[~, SortOrder] = sort([pop.Cost]);
pop = pop(SortOrder);
% Update Best Solution Ever Found
BestSol = pop(1);
% Store Best Cost Ever Found
BestCost(it) = BestSol.Cost;
% Show Iteration Information
disp(['In Iteration ' num2str(it) ' : BBO-FireFly Fittest Cost Is =  ' num2str(BestCost(it))]);
end
%------------------------------------------------
% disp('BBO Algorithm Came To End');

% Store Result
results.BestSol=BestSol;
results.BestCost=BestCost;
% Plot BBO Algorithm Training Stages
figure;
set(gcf, 'Position',  [600, 300, 500, 300])
plot(BestCost,'-',...
'LineWidth',2,...
'MarkerSize',3,...
'MarkerEdgeColor','k',...
'Color',[0.1,0.1,0.1]);
title('BBO Algorithm Training','FontSize',10,...
'FontWeight','bold','Color','b');
xlabel('BBO Iteration Number','FontSize',10,...
'FontWeight','bold','Color','k');
ylabel('BBO Best Cost Result','FontSize',10,...
'FontWeight','bold','Color','k');
legend({'BBO Algorithm Train'});
end






