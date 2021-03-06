model = gallery_mm1;
sa = SolverJMT(model).sampleSysAggr(5e4);

eventarr = cellfun(@(c) c.node == 2 && c.event == EventType.ARV, sa.event);
arvtime = cellfun(@(c) (c.t), {sa.event{eventarr}});
arvstates = zeros(length(arvtime), 1);

for i = 1:length(arvtime)
    arvstates(i) = sa.state{2}(sa.t == (arvtime(i))) - 1;
end

eventdep = cellfun(@(c) c.node == 2 && c.event == EventType.DEP, sa.event);
deptime = cellfun(@(c) (c.t), {sa.event{eventdep}});

resptime = deptime - arvtime(1:length(deptime));
m = mean(resptime);
n = SolverJMT(model).getAvgTable;
disp(m)
disp(n)