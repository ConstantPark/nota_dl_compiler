import tvm
from tvm import relay
import numpy as np

data = relay.var("data", shape=(1, 1, 16, 16), dtype="float32")
weight = relay.const(np.random.uniform(size=(1, 1, 1, 1)))

conv = relay.nn.conv2d(data, weight)
model = relay.nn.relu(conv)

args = relay.ir_pass.free_vars(model)
func = relay.Function(args, model) 

print(func) 

target = "cuda"
context = tvm.gpu(0)

graph, lib, params = relay.build(func, target=target)
module = tvm.contrib.graph_runtime.create(graph, lib, context)
module.set_input(**params)
module.run()
