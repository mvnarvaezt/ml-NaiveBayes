% Clasificador bayesiano con PDF Gaussiana y regla MAP
% Author: Valentina Narvaez  

clear all                                      

choise = input('Clasificador Bayesiano con PDF Gausiana y seleccio?n MAP\n Elige dataset\n 1: overlapped.mat\n2: clouds01.mat\n3: clouds02.mat\n4: twospirals.mat\n5: halfkernel.mat\nElge un dataset:');

switch choise 
    case 1
        dataset = '/datasets/overlapped.mat';
        setName = 'overlaped';
    case 2
        dataset = '/datasets/clouds01.mat';
        setName = 'clouds01';
    case 3
        dataset = '/datasets/clouds02.mat';
        setName = 'clouds02';
    case 4
        dataset = '/datasets/twospirals.mat';
        setName = 'twoSpirals';
    case 5
        dataset = '/datasets/halfkernel.mat';
        setName = 'halfKernel';
    otherwise
        disp('No existe.');
end


load([pwd dataset]);  % x atributos, y variable de clase
runs = 100;
error = zeros(1,runs);

for r = 1 : runs
    cvp = cvpartition(Y, 'Holdout', 0.3);    % vectores logicos para entrenamiento y prueba
    %[train,test] = crossvalind('HoldOut',Y,0.3);    
    train = cvp.training;
    test = ~train;
    xTrain = X(:, train);                    % Xtrain: conjunto entrenamiento
    yTrain = Y(:, train);                    % Ytrain: variable de clase del conjunto de entrenamiento

    xTest = X(:, test);                      % conjunto de prueba
    yTest = Y(:, test);                      % variable de clase del conjunto de prueba

    [mu_i,sigma_i,p_i] = trainBayes(xTrain, yTrain);                        % Entrenamiento
    [yPred, error(:,r)] = classifyBayes(xTest, yTest,mu_i, sigma_i, p_i);   % Prueba
    nClasses = length(unique(yPred));
    
end

res = 400;                      % Generacion de patrones para pintar el fondo
xMin = min(xTest, [], 2);
xMax = max(xTest, [], 2);
[x1, x2] = meshgrid(linspace(xMin(1), xMax(1),res), linspace(xMin(2), xMax(2),res));
x = [x1(:),x2(:)]';
y = classifyBayes(x, zeros, mu_i, sigma_i, p_i);  % Clasificacion de los patrones del fondo

%%
f1 = figure
title('Error rate')
plot(error,'-r')

%%
f2 = figure;
title( strcat('Data set: ', setName) )
hold on
for c = 1: nClasses                         % Por cada clase c
    myplot = x(:,y == c);                   % Otiene los patrones de la clase
   h1 = scatter(myplot(1,:),myplot(2,:),'.');      % Grafica la clase obtenida
   h1.Annotation.LegendInformation.IconDisplayStyle = 'off';
   alpha(h1,0.0);
end

for c = 1: nClasses                         % Por cada clase c
    myplot = xTest(:,yPred == c);           % Otiene los patrones de la clase
    h2 = scatter(myplot(1,:),myplot(2,:), 5, 'ob', 'MarkerFaceColor','w' );      % Grafica la clase obtenida
    h2.Annotation.LegendInformation.IconDisplayStyle = 'off';
end
h2.Annotation.LegendInformation.IconDisplayStyle = 'on';
myplot = xTest(:,yPred ~= yTest);            % Otiene los patrones con error
scatter(myplot(1,:),myplot(2,:),15,'or', 'MarkerFaceColor','w');      % Grafica los errores

xlabel('Pred 1');
ylabel('Pred 2');
legend({'correct', 'errors'});

sum(error)/runs

print(f1, strcat('img/',setName,'-error'), '-dpng' );
print(f2, strcat('img/',setName,'-class'), '-dpng' );

