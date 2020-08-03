[model,source,queue,sink,oclass]=gallery_merl1;
solver = SolverCTMC(model,'cutoff',100,'seed',23000);
sa = solver.sampleSysAggr(5e4);

eventarr = cellfun(@(c) c.node == 2 && c.event == EventType.ARV, sa.event);
arvtime = cellfun(@(c) (c.t), {sa.event{eventarr}});
arvstates = zeros(length(arvtime), 1);

for i = 1:length(arvtime)
    arvstates(i) = sa.state{2}(sa.t == (arvtime(i)));
end

plot(arvtime(1:500), arvstates(1:500))
xlabel('time');
ylabel('queue length');
title('States seen on arriving');