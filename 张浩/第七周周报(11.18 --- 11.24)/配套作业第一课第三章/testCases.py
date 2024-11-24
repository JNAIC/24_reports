import numpy as np

def layer_sizes_test_case():
    np.random.seed(0)
    X_assess = np.random.randn(2, 3)
    Y_assess = np.random.randn(1, 3)
    # print("hello world")
    return X_assess, Y_assess

def initialize_parameters_test_case():
    n_x, n_h, n_y = 2, 4, 1
    return n_x, n_h, n_y

def forward_propagation_test_case():
    np.random.seed(0)
    X_assess = np.random.randn(2, 3)

    parameters = {'W_1': np.array([[-0.00416758, -0.00056267],
        [-0.02136196,  0.01640271],
        [-0.01793436, -0.00841747],
        [ 0.00502881, -0.01245288]]),
     'W_2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'b_1': np.array([[ 0.],
        [ 0.],
        [ 0.],
        [ 0.]]),
     'b_2': np.array([[ 0.]])}

    return X_assess

def compute_cost_test_case():
    np.random.seed(0)
    Y_assess = np.random.randn(1, 3)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
        [-0.02136196,  0.01640271],
        [-0.01793436, -0.00841747],
        [ 0.00502881, -0.01245288]]),
     'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'b1': np.array([[ 0.],
        [ 0.],
        [ 0.],
        [ 0.]]),
     'b2': np.array([[ 0.]])}
    return Y_assess

def backward_propagation_test_case():
    np.random.seed(0)
    X_assess = abs(np.random.randn(2, 2))
    Y_assess = np.random.randn(1, 2)
    parameters = {'W_1': np.array([[-0.00416758, -0.00056267],
        [-0.02136196,  0.01640271],
        [-0.01793436, -0.00841747],
        [ 0.00502881, -0.01245288]]),
     'W_2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'b_1': np.array([[ 0.],
        [ 0.],
        [ 0.],
        [ 0.]]),
     'b_2': np.array([[ 0.]])}
    
    return X_assess, Y_assess, parameters


np.random.seed(3)
print(np.random.randn(3))
    
