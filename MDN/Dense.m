classdef Dense < handle
    %Dense layer
    
    properties
        layer_input
        input_shape
        n_units
        W
        w0
    end
    
    methods
        function obj = Dense(n_units,input_shape)
            obj.layer_input = [];
            obj.input_shape = input_shape;
            obj.n_units = n_units;
            obj.W = [];
            obj.w0 = [];
        end
        
        function set_input_shape(obj, shape)
            obj.input_shape = shape;
        end
        
        function Y = output_shape(obj)
            Y = [obj.n_units];
        end
        
        function initialize(obj)
            limit = 1/sqrt(obj.input_shape(1));
            obj.W = (2*limit).*rand(obj.input_shape(1), obj.n_units) - limit;
            obj.w0 = zeros(1, obj.n_units);
        end
        
        function Y = forward_pass(obj, X)
            obj.layer_input = X;
            Y = X*(obj.W) + obj.w0;
        end
        
        function accum_grad = backward_pass(obj, accum_grad)
            Wi = obj.W;
            grad_w = ((self.layer_input)')*accum_grad;
            grad_w0 = sum(accum_grad);
            
            %adam optimizer
            learning_rate = 0.001;
            b1 = 0.9;
            b2 = 0.999;
            eps = 1e-8;
            
            m = zeros(size(grad_w));
            v = zeros(size(grad_w));
            m = b1*m*(1-b1)*grad_w;
            v = b2*v*(1-b2)*(grad_w.^2);
            m_hat = m/(1-b1);
            v_hat = v/(1-b2);
            w_updt = learning_rate*m_hat/(sqrt(v_hat) + eps);
            obj.W = obj.W - w_updt;
            
            m = zeros(size(grad_w0));
            v = zeros(size(grad_w0));
            m = b1*m*(1-b1)*grad_w0;
            v = b2*v*(1-b2)*(grad_w0.^2);
            m_hat = m/(1-b1);
            v_hat = v/(1-b2);
            w_updt = learning_rate*m_hat/(sqrt(v_hat) + eps);
            obj.w0 = obj.w0 - w_updt;
            
            accum_grad = accum_grad*Wi';
        end
            
    end
end

