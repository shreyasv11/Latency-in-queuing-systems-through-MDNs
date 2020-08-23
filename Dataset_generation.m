[ti] = NonHomogPP(100, 2, 5, 1);

model = Network('myModel');
source = Source(model, 'mySource');
queue = Queue(model, 'myQueue', SchedStrategy.FCFS);
queue.setNumServers(3);
sink = Sink(model, 'mySink');
oclass = OpenClass(model, 'myClass');
source.setArrival(oclass, Exp(1));
queue.setService(oclass, Replayer('trace.txt'));
model.link(Network.serialRouting(source,queue,sink));

sa = SolverJMT(model).sampleSysAggr(5e4);
eventarr = cellfun(@(c) c.node == 2 && c.event == EventType.ARV, sa.event);
arvtime = cellfun(@(c) (c.t), {sa.event{eventarr}});
arvstates = zeros(length(arvtime), 1);

for i = 1:length(arvtime)
    arvstates(i) = sa.state{2}(sa.t == (arvtime(i))) - 1;
end

eventdep = cellfun(@(c) c.node == 2 && c.event == EventType.DEP, sa.event);
dtime = cellfun(@(c) (c.t), {sa.event{eventdep}});

arv_ids = sa.arv_job_id;
dep_ids = sa.dep_job_id;
arvtime_new = zeros(1, length(dtime));

for i=1:length(dep_ids)
    k = (arv_ids == dep_ids(i));
    arvtime_new(i) = arvtime(k);
end

resptime = dtime - arvtime_new;
m = mean(resptime);
n = SolverJMT(model).getAvgTable;
disp(m)
disp(n)