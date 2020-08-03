model = gallery_mm1;
sa = SolverJMT(model).sampleSysAggr(5e4);

eventarr = cellfun(@(c) c.node == 2 && c.event == EventType.ARV, sa.event);
arvtime = cellfun(@(c) (c.t), {sa.event{eventarr}});
arvstates = zeros(length(arvtime), 1);

for i = 1:length(arvtime)
    arvstates(i) = sa.state{2}(sa.t == (arvtime(i))) - 1;
end

eventdep = cellfun(@(c) c.node == 2 && c.event == EventType.DEP, sa.event);
dtime = cellfun(@(c) (c.t), {sa.event{eventdep}});
deptime = zeros(1, length(dtime));

arv_ids = sa.arv_job_id;
dep_ids = sa.dep_job_id;

for i=1:length(dep_ids)
    k = (arv_ids == dep_ids(i));
    deptime(i) = dtime(k);
end

resptime = deptime - arvtime(1:length(deptime));
m = mean(resptime);
n = SolverJMT(model).getAvgTable;
disp(m)
disp(n)