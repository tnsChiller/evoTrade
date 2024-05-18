for i in range(10):
    x = np.random.randint(0, 100)
    y = np.random.randint(0, 98)
    z = np.random.randint(0, 3483)
    a = scr[x, y, z]
    b = thr[x, y, 0]
    c = c0[x, y, z]
    print(f"{a} > {b}, {c}")