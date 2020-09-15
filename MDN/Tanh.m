classdef Tanh < handle
    %Tanh layer
    
    properties
        activation_name
        layer_input
        input_shape
    end
    
    methods
        function obj = Tanh(s)
            obj.activation_name = s;
        end
        
        function set_input_shape(obj, shape)
            obj.input_shape = shape;
        end
        
        function Y = output_shape(obj)
            Y = obj.input_shape;
        end
        
        function y = tanh_m(obj, x)
            y = 2./(1 + exp(-2*x)) - 1;
        end
        
        function Y = forward_pass(obj,X)
            obj.layer_input = X;
            Y = obj.tanh_m(X);
        end
        
        function accum_grad = backward_pass(obj, accum_grad)
            accum_grad = accum_grad .* (1 - ((obj.tanh_m(obj.layer_input)).^2));
        end
    end
end

