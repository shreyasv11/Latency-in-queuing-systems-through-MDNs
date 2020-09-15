classdef MDN_Layer < handle
    %MDN Layer
    
    properties
        input_shape
        input_dim
        output_dim
        num_components
        activation_sigma
        layer_input
        activation_pi
        Wmu
        Wmu_opt
        bmu
        bmu_opt
        Wsigma
        Wsigma_opt
        bsigma
        bsigma_opt
        Wpi
        Wpi_opt
        bpi
        bpi_opt
        sigma
        pii
    end
    
    methods
        function obj = MDN_Layer(num_components, input_shape, output_shape)
            obj.input_shape = input_shape;
            obj.output_dim = output_shape;
            obj.num_components = num_components;
        end
        
        function set_input_shape(obj, shape)
            obj.input_shape = shape;
        end
        
        function Y = output_shape(obj)
            Y = [obj.output_dim * 3];
        end
        
        function initialize(obj)
            obj.input_dim = obj.input_shape(1);
            obj.output_dim = obj.num_components;
            obj.Wmu = randn(obj.input_dim, obj.output_dim) * 0.1;
            obj.bmu = randn(1, obj.output_dim) * 0.01;
            obj.Wsigma = randn(obj.input_dim, obj.output_dim) * 0.1;
            obj.bsigma = randn(1, obj.output_dim) * 0.01;
            obj.Wpi = randn(obj.input_dim, obj.output_dim) * 0.1;
            obj.bpi = randn(1, obj.output_dim) * 0.01;
            obj.Wmu_opt = AdamOptimizer(); obj.bmu_opt = AdamOptimizer();
            obj.Wpi_opt = AdamOptimizer(); obj.bpi_opt = AdamOptimizer();
            obj.Wsigma_opt = AdamOptimizer(); obj.bsigma_opt = AdamOptimizer();
        end
        
        function y = Softmax_m(obj, x)
            e_x = exp(x - max(x, [], 2));
            y = e_x ./ sum(e_x, 2);
        end
        
        function Y = forward_pass(obj, X)
            obj.layer_input = X;
            mu = obj.layer_input*obj.Wmu + obj.bmu;
            obj.sigma = exp((obj.layer_input*obj.Wsigma) + obj.bsigma);
            obj.pii = obj.Softmax_m((obj.layer_input*obj.Wpi) + obj.bpi);
            Y = {obj.pii, mu, obj.sigma};
        end
        
        function accum_grad = backward_pass(obj, accum_grad)
            Wpii = obj.Wpi;
            Wmui = obj.Wmu;
            Wsigmai = obj.Wsigma;
            dpi = accum_grad{1}; dmu = accum_grad{2}; dsigma = accum_grad{3};
            dpi = dpi .* ((obj.Softmax_m(obj.pii).*(1 - obj.Softmax_m(obj.pii))));
            grad_Wpi = ((obj.layer_input)')*dpi;
            grad_bpi = sum(dpi);
            grad_Wmu = ((obj.layer_input)')*dmu;
            grad_bmu = sum(dmu);
            dsigma = dsigma .* (exp(obj.sigma));
            grad_Wsigma = ((obj.layer_input)')*dsigma;
            grad_bsigma = sum(dsigma);
            
            obj.Wpi = obj.Wpi_opt.update(obj.Wpi, grad_Wpi);
            obj.bpi = obj.bpi_opt.update(obj.bpi, grad_bpi);
            obj.Wmu = obj.Wmu_opt.update(obj.Wmu, grad_Wmu);
            obj.bmu = obj.bmu_opt.update(obj.bmu, grad_bmu);
            obj.Wsigma = obj.Wsigma_opt.update(obj.Wsigma, grad_Wsigma);
            obj.bsigma = obj.bsigma_opt.update(obj.bsigma, grad_bsigma);
            
            accum_grad = (dpi*Wpii') + (dmu*Wmui') + (dsigma*Wsigmai');
        end
    end
end

