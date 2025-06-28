import numpy as np

def get_function_details(function_number):
    """
    Get the details of test functions including function handle, number of variables,
    and bounds
    """
    functions = {
        1: (f1, 2, -4.5 * np.ones(2), 4.5 * np.ones(2)),
        2: (f2, 2, -10 * np.ones(2), 10 * np.ones(2)),
        3: (f3, 2, -5 * np.ones(2), 5 * np.ones(2)),
        4: (f4, 30, -1 * np.ones(30), 1 * np.ones(30)),
        5: (f5, 30, -5 * np.ones(30), 5 * np.ones(30)),
        6: (f6, 30, -100 * np.ones(30), 100 * np.ones(30)),
        7: (f7, 30, -5.12 * np.ones(30), 5.12 * np.ones(30)),
        8: (f8, 2, -5.2 * np.ones(2), 5.2 * np.ones(2)),
        9: (f9, 2, -512 * np.ones(2), 512 * np.ones(2)),
        10: (f10, 2, -5 * np.ones(2), 5 * np.ones(2))
    }
    
    if function_number not in functions:
        function_number = 1
    
    return functions[function_number]

def f1(x):
    """Beale function"""
    f1 = (1.5 - x[:, 0] + x[:, 0] * x[:, 1])**2
    f2 = (2.25 - x[:, 0] + x[:, 0] * x[:, 1]**2)**2
    f3 = (2.625 - x[:, 0] + x[:, 0] * x[:, 1]**3)**2
    return f1 + f2 + f3

def f2(x):
    """Booth function"""
    return 0.26 * np.sum(x**2, axis=1) - 0.48 * np.prod(x, axis=1)

def f3(x):
    """Three-hump camel function"""
    return (2 * x[:, 0]**2 - 1.05 * x[:, 0]**4 + (x[:, 0]**6) / 6 + 
            x[:, 0] * x[:, 1] + x[:, 1]**2)

def f4(x):
    """Exponential function"""
    return -np.exp(-0.5 * np.sum(x**2, axis=1))

def f5(x):
    """Zakharov function"""
    d = 2
    alpha = 0.1
    return x[:, 0] + d * (np.sum(x[:, 1:]**2, axis=1)**alpha)

def f6(x):
    """Sphere function"""
    return np.sum(x**2, axis=1)

def f7(x):
    """Shifted sphere function"""
    return np.sum((x + 0.5)**2, axis=1)

def f8(x):
    """Drop-wave function"""
    return -(1 + np.cos(12 * np.sqrt(np.sum(x**2, axis=1)))) / (0.5 * np.sum(x**2, axis=1) + 2)

def f9(x):
    """Schaffer N.6 function"""
    return np.sum(-(x[:, 1] + 47) * np.sin(np.sqrt(np.abs(x[:, 1] + x[:, 0]/2 + 47))) - 
                 x[:, 0] * np.sin(np.sqrt(np.abs(x[:, 0] - (x[:, 1] + 47)))), axis=1)

def f10(x):
    """Himmelblau function"""
    return ((x[:, 0]**2 + x[:, 1] - 11)**2 + 
            (x[:, 0] + x[:, 1]**2 - 7)**2) 