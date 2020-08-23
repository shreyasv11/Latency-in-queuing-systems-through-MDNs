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
        bmu
        Wsigma
        bsigma
        Wpi
        bpi
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
            obj.bmu = randn(obj.output_dim, 1) * 0.01;
            obj.Wsigma = randn(obj.input_dim, obj.output_dim) * 0.1;
            obj.bsigma = randn(obj.output_dim, 1) * 0.01;
            obj.Wpi = randn(obj.input_dim, obj.output_dim) * 0.1;
            obj.bpi = randn(obj.output_dim, 1) * 0.01;
        end
        
        function Y = forward_pass(obj, X)
            obj.layer_input = X;
            mu = obj.layer_input*obj.Wmu + obj.bmu;
            obj.sigma = exp((obj.layer_input*obj.Wsigma) + obj.bsigma);
            obj.pii = softmax((obj.layer_input*obj.Wpi) + obj.bpi);
            Y = {obj.pii, mu, obj.sigma};
        end
        
        function accum_grad = backward_pass(obj, accum_grad)
            Wpii = obj.Wpi;
            Wmui = obj.Wmu;
            Wsigmai = obj.Wsigma;
            
            dpi = accum_grad{1}; dmu = accum_grad{2}; dsigma = accum_grad{3};
            dpi = dpi * (softmax(self.pii)*(1 - softmax(self.pii)));
            grad_Wpi = ((obj.layer_input)')*dpi;
            grad_bpi = sum(dpi);
            grad_Wmu = ((obj.layer_input)')*dmu;
            grad_bmu = sum(dmu);
            dsigma = dsigma*(exp(obj.sigma));
            grad_Wsigma = ((obj.layer_input)')*dsigma;
            grad_bsigma = sum(dsigma);
            
            obj.Wpi = obj.update(obj.Wpi, grad_Wpi);
            obj.bpi = obj.update(obj.bpi, grad_bpi);
            obj.Wmu = obj.update(obj.Wmu, grad_Wmu);
            obj.bmu = obj.update(obj.bmu, grad_bmu);
            obj.Wsigma = obj.update(obj.Wsigma, grad_Wsigma);
            obj.bsigma = obj.update(obj.bsigma, grad_bsigma);
            
            accum_grad = (dpi*Wpii') + (dmu*Wmui') + (dsigma*Wsigmai');
        end
        
        function wt = update(wt, grad_wrt_w)
            m = zeros(size(grad_wrt_w));
            v = zeros(size(grad_wrt_w));
            m = b1*m*(1-b1)*grad_wrt_w;
            v = b2*v*(1-b2)*(grad_wrt_w.^2);
            m_hat = m/(1-b1);
            v_hat = v/(1-b2);
            w_updt = learning_rate*m_hat/(sqrt(v_hat) + eps);
            wt = wt - w_updt;
        end
    end
end

