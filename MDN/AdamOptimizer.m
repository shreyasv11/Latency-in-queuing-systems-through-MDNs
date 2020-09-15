classdef AdamOptimizer < handle
    %Adam optimizer function
    
    properties
        learning_rate
        epsi
        m
        v
        b1
        b2
    end
    
    methods
        function obj = AdamOptimizer()
            obj.learning_rate = 0.001;
            obj.b1 = 0.9;
            obj.b2 = 0.999;
            obj.epsi = 1e-8;
            obj.m = -1;
        end
        
        function wt = update(obj, wt, grad_wrt_w)
            if obj.m == -1
                obj.m = zeros(size(grad_wrt_w));
                obj.v = zeros(size(grad_wrt_w));
            end
            obj.m = obj.b1*obj.m + (1-obj.b1)*grad_wrt_w;
            obj.v = obj.b2*obj.v + (1-obj.b2)*(grad_wrt_w.^2);
            m_hat = obj.m./(1-obj.b1);
            v_hat = obj.v./(1-obj.b2);
            w_updt = obj.learning_rate*m_hat./(sqrt(v_hat) + obj.epsi);
            wt = wt - w_updt;
        end
    end
end

