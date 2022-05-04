% Clasificador bayesiano con PDF Gaussiana y regla MAP
% Author: Valentina Narvaez  

clear all                                        % pwd es la carpeta actual

choise = input('Clasificador Bayesiano con PDF Gausiana y seleccioón MAP\n Elige dataset\n 1: overlapped.mat\n2: clouds01.mat\n3: clouds02.mat\n4: twospirals.mat\n5: halfkernel.mat\nElge un dataset:');

switch choise 
    case 1
        dataset = '/datasets/overlapped.mat';
    case 2
        dataset = '/datasets/clouds01.mat';
    case 3
        dataset = '/datasets/clouds02.mat';
    case 4
        dataset = '/datasets/twospirals.mat';
    case 5
        dataset = '/datasets/halfkernel.mat';
    otherwise
        disp('No existe.');
end


load([pwd dataset]);  % x atributos, y variable de clase
runs = 31;
error = zeros(1,30);

for r = 1 : runs
    [train,test] = crossvalind('HoldOut',Y,0.3);    % vectores logicos para entrenamiento y prueba
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

figure
plot(error,'-r')

figure;
hold all
for c = 1: nClasses                         % Por cada clase c
    myplot = x(:,y == c);                   % Otiene los patrones de la clase
    plot(myplot(1,:),myplot(2,:),'.');      % Grafica la clase obtenida
end
for c = 1: nClasses                         % Por cada clase c
    myplot = xTest(:,yPred == c);           % Otiene los patrones de la clase
    plot(myplot(1,:),myplot(2,:),'.');      % Grafica la clase obtenida
end
myplot = xTest(:,yPred ~= yTest);            % Otiene los patrones con error
plot(myplot(1,:),myplot(2,:),'+b');      % Grafica los errores

sum(error)/31