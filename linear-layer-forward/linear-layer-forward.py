def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    X_m = len(X)
    X_n = len(X[0])
    W_n = len(W[0])
    res = [[0] * W_n for _ in range(X_m)] # (X_m, W_n)
    for x_row_ind in range(X_m):
        for w_col_ind in range(W_n):
            acc = 0
            for shared_ind in range(X_n):
                acc += X[x_row_ind][shared_ind] * W[shared_ind][w_col_ind]
            res[x_row_ind][w_col_ind] = acc
    for i in range(len(res)):
        for j in range(len(res[i])):
            res[i][j] += b[j]
    return res