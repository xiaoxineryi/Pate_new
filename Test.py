import Input
import numpy as np
test_data,test_labels = Input.load_mnist(test_only=True)
test_data = test_data[:1000]
test_data = np.delete(test_data,[1,2,3],axis=0)

print(len(test_data))