% Clasificador bayesiano: Entrenamiento

function [mu_i, sigma_i, p_i] = trainBayes(xTrain, yTrain)
    [ndimTrain, nTrain] = size(xTrain);      % numero de atributos e instancias en el conj. de entrenamiento
    classes = unique(yTrain);                % Conjunto de valores posibles de la variable de clase  
    nClasses = numel(classes);               % Numero de clases
    mu_i = zeros(ndimTrain, nClasses);       % 
    sigma_i = zeros(ndimTrain, ndimTrain, nClasses);   % una matriz por clase
    p_i = zeros(1, nClasses);               % Probabilidades de cada clase

    for i = 1 : nClasses                    % Por cada clase..
        wi = yTrain == i;                   % Instancias de la clase i 
        mu_i(:,i) = mean(xTrain(:, wi),ndimTrain);       % Media por dimension, para c/clase i
        sigma_i(:,:,i) = cov(xTrain(:,wi)', 0);          % Matrices de covarianza, por clase
        p_i(i) = sum(wi) / nTrain;
    end
end