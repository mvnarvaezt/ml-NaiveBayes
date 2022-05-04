% Clasificador bayesiano: Clasificacion

function [rClass, error] = classifyBayes(xTest, yTest, mu_i, sigma_i, p_i)
    [ndim, n] = size(xTest);                            % Dimensiones, numero de instancias
    [ndimTest, nTest] = size(xTest);                    
    nClasses = size(mu_i, ndim);
    inverse_sigma_i = zeros(ndim, ndim, nClasses);
    c_i = zeros(1, nClasses);
    yPred = zeros(1,nTest);

    for i = 1 : nClasses
       inverse_sigma_i(:,:,i) = pinv(sigma_i(:,:,i));   % Inversa de i
       c_i(i) = 1 / ((2 * pi)^(ndimTest/2)) * sqrt(det(sigma_i(:,:,i)));   % PDF (seccion independiente)
    end

    rClass = zeros (size(yTest));
    for k = 1 : nTest
        pwx = zeros (1, nClasses);
        for i = 1 : nClasses                
           xmu = xTest(:,k) - mu_i(:,i);    % Atributos del patron menos las medias
           pwx(i) = c_i(i) * exp( (-0.5 * xmu') * (inverse_sigma_i(:,:,i) * xmu) ); % evaluacion de PDF 
        end
        pwx(:) = pwx(:) .* p_i(:);              % Ponderacion con la probabilidad de la clase
        rClass(k) = find(pwx == max(pwx(:)));   % Desicion MAP
    end

    Q = yTest ~= rClass;                        % Diferencias entre clase real y la predicha
    error = (100 / length(xTest)) * sum(Q);     % Calculo del error
end

