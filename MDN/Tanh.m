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
        
        function Y = forward_pass(obj,X)
            obj.layer_input = X;
            Y = tanh(X);
        end
        
        function accum_grad = backward_pass(obj, accum_grad)
            accum_grad = accum_grad * (1 - (tanh(obj.layer_input)).^2);
        end
    end
end

