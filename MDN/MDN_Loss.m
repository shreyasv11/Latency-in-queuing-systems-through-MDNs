classdef MDN_Loss < handle
    %MDN Loss function
    
    properties
        mixtures
        output
        eps
        ypred
    end
    
    methods
        function obj = MDN_Loss(num_components, output_dim)
            obj.mixtures = num_components;
            obj.output = output_dim;
            obj.eps = 1e-5;
            obj.ypred = [];
        end
        
        function loss = loss(obj, pii, sigma, mu, y_true)
            result = normpdf(y_true, mu, sigma) * pii;
            result = sum(result);
            result = -log(result + obj.eps);
            obj.ypred = result;
            loss = mean(result);
        end
        
        function as = acc(obj, y_true)
            as = sum(y_true == obj.y_pred);
        end
        
        function [dpi, dmu, dsigma] = gradient(obj, pii, sigma, mu, y_true)
            N = size(y_true, 1);
            g = normpdf(y_true, mu, sigma) * pii;
            gamma = g/sum(g);
            dmu = gamma*((mu - y_true)/sigma.^2);
            dmu = dmu/N;
            dsigma = gamma*(1 - ((y_true - mu).^2)/sigma.^2);
            dsigma = dsigma/N;
            dpi = (pii - gamma)/N;
        end
    end
end

