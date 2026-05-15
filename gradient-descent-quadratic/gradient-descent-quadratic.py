def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    x = x0
    for i in range(steps):
        z = a * x * x + b * x + c

        # dz / dx = 2ax + b
        dx = 2 * a * x + b

        x -= lr * dx
    return x