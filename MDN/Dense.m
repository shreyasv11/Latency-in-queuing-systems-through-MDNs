classdef Dense < handle
    %Dense layer
    
    properties
        layer_input
        input_shape
        n_units
        W
        w0
        W_opt
        w0_opt
    end
    
    methods
        function obj = Dense(n_units,input_shape)
            obj.layer_input = [];
            obj.input_shape = input_shape;
            obj.n_units = n_units;
            obj.W = [];
            obj.w0 = [];
            obj.W_opt = AdamOptimizer();
            obj.w0_opt = AdamOptimizer();
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
            grad_w = ((obj.layer_input)')*accum_grad;
            grad_w0 = sum(accum_grad);
            
            obj.W = obj.W_opt.update(obj.W, grad_w);
            obj.w0 = obj.w0_opt.update(obj.w0, grad_w0);
            
            accum_grad = accum_grad*Wi';
        end
            
    end
end

