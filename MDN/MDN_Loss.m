classdef MDN_Loss < handle
    %MDN Loss function
    
    properties
        mixtures
        output
        epsi
        ypred
    end
    
    methods
        function obj = MDN_Loss(num_components, output_dim)
            obj.mixtures = num_components;
            obj.output = output_dim;
            obj.epsi = 1e-5;
            obj.ypred = [];
        end
        
        function k = gaussian_pdf(obj, x, mu, sigma)
            d = sqrt(2*pi) * sigma + obj.epsi;
            n = exp(-((x - mu).^2)./(2 * (sigma.^2)));
            k = n./d;
        end
        
        function loss = loss(obj, pii, sigma, mu, y_true)
            result = obj.gaussian_pdf(y_true, mu, sigma) .* pii;
            result = sum(result, 2);
            result = -log(result + obj.epsi);
            obj.ypred = result;
            loss = mean(result);
        end
        
        function [dpi, dmu, dsigma] = gradient(obj, pii, sigma, mu, y_true)
            N = size(y_true, 1);
            g = obj.gaussian_pdf(y_true, mu, sigma) .* pii;
            gamma = g./sum(g,2);
            dmu = gamma .* ((mu - y_true)./(sigma.^2));
            dmu = dmu/N;
            dsigma = gamma .* (1 - ((y_true - mu).^2)./(sigma.^2));
            dsigma = dsigma/N;
            dpi = (pii - gamma)/N;
        end
    end
end

