# 2020.12.01
# Created by jinlj

from . import tfcompat
from . import backend

__version__ = "2020.12.01 (by jinlj; onnx 1.8; pytorch 1.7)"

def onnx2tf(onnxfpath, pbfpath=""):
    import onnx

    #(1) 加载 onnx 模型
    onnx_model = onnx.load(onnxfpath)

    #(2) 转换 onnx 到 pb 格式
    tf_rep = backend.prepare(onnx_model,
                device='GPU',
                strict=True,
                logging_level='INFO')

    #(3) 保存 pb 模型文件
    if not pbfpath:
        pbfpath = onnxfpath[:onnxfpath.rfind(".")] + ".pb"
    tf_rep.export_graph(pbfpath)

def runTf(pbfpath, feed_dict, outputs, loops=10):
    """Take Yolov5 for example:
    pbfpath：exported pb file path.
    feed_dict: {"input.1:0": xxx}
    outputs: ["Add_64:0", "Add_57:0", "Add_49:0"]
    """
    import tensorflow as tf
    import time, sys, os

    if not os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 1.0 # 程序最多只能占用的显存
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default() as graph:
        graph_def = tf.GraphDef()
        sess = tf.Session(graph=graph, config=config)
        init = tf.global_variables_initializer()
        with open(pbfpath, "rb") as fin:
            graph_def.ParseFromString(fin.read())
            _=tf.import_graph_def(graph_def, name="")

        
        
        sess.run(init)

        fetches = [sess.graph.get_tensor_by_name(x) for x in outputs]

        print("\n=> Check time (forward only)...\n")
        ts = time.time()
        preds = sess.run(fetches=fetches, feed_dict=feed_dict)
        print("=> first run cost: {:.3f} ms".format( (time.time() - ts)*1000 ) )

        ts = time.time()
        for i in range(loops):
            preds = sess.run(fetches=fetches, feed_dict=feed_dict)

        dt = (time.time() - ts)*1000
        print("=> next {} loops cost {:.3f} ms".format( loops, dt))
        print("=> average {:.3f} ms ".format( dt/loops ) )

        return preds