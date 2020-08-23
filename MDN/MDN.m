components = 3;
loss = MdnLoss(components, 1);
layers = {};

dense = Dense(26, [1]);
dense.initialize();
layer{1} = dense;

tanh_layer = Tanh('tanh');
tanh_layer.set_input_shape(layer{1}.output_shape());
layer{2} = tanh_layer;

mdn_layer = MDN_Layer([26], [1]);
mdn_layer.set_input_shape(layer{2}.output_shape());
mdn_layer.initialize();
layer{3} = mdn_layer;

n = 225;
d = 1;
x = rand(n, d);
noise = (0.2).*rand(n,d) - 0.1;
y = x + 0.3*sin(2*pi*x) + noise;

