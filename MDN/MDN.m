rand('twister',2);
randn('seed',2);
components = 3;
loss_function = MDN_Loss(components, 1);
layers = {};

dense = Dense(26, [1]);
dense.initialize();
layers{1} = dense;

tanh_layer = Tanh('tanh');
tanh_layer.set_input_shape(layers{1}.output_shape());
layers{2} = tanh_layer;

mdn_layer = MDN_Layer(components, [26], [1]);
mdn_layer.set_input_shape(layers{2}.output_shape());
mdn_layer.initialize();
layers{3} = mdn_layer;

n = 250;
d = 1;
y = rand(n,d);
noise = (0.2).*rand(n,d) - 0.1;
x = y + 0.3*sin(2*pi*y) + noise;
epochs = 1;
error = zeros(epochs, 1);

for i=1:epochs
    [y_pred, layers] = forward_run(x, layers);
    pii = y_pred{1};
    mu = y_pred{2};
    sigma = y_pred{3};
    loss = loss_function.loss(pii, sigma, mu, y);
    error(i) = loss;
    [dpi, dmu, dsigma] = loss_function.gradient(pii, sigma, mu, y);
    loss_grad = {};
    loss_grad{1} = dpi;
    loss_grad{2} = dmu;
    loss_grad{3} = dsigma;
    layers = backward_run(loss_grad, layers);
end
figure(2)
plot(error)

n_test = 250;
% y_test = rand(n_test,d);
% noise_test = (0.2).*rand(n_test,d) - 0.1;
% x_test = y_test + 0.3*sin(2*pi*y_test) + noise_test;
x_test = linspace(0,1,n_test);
x_test = transpose(x_test);
% [result,layers] = forward_run(x_test, layers);
% y_test = generate_ensemble(result);

function t = get_pi_idx(x, pdf)
    N = size(pdf,2);
    accumulate = 0;
    t = -1;
    for i=1:N
        accumulate = accumulate + pdf(i);
        if accumulate >= x
            t = i;
        end
    end
end

function result = generate_ensemble(y_test)
    M = 1;
    out_pi = y_test{1}; out_mu = y_test{2}; out_sigma = y_test{3};
    ntest = size(out_pi, 1);
    result = rand(ntest, M);
    rn = randn(ntest, M);
    mu = 0;
    std = 0;
    idx = 0;
    for j=1:M
        for i=1:ntest
            idx = get_pi_idx(result(i, j),out_pi(i,:));
            mu = out_mu(i, idx);
            std = out_sigma(i, idx);
            result(i, j) = mu + rn(i, j)*std;
        end
    end
end

function [output, layers] = forward_run(X, layers)
    output = X;
    for i=1:length(layers)
        curlayer = layers{i};
        output = curlayer.forward_pass(output);
    end
end

function layers = backward_run(loss_grad, layers)
    for i=length(layers):-1:1
        curlayer = layers{i};
        loss_grad = curlayer.backward_pass(loss_grad);
    end
end