
p, m = 100, 35
a = np.random.rand(p, m)
s = np.random.rand(p, 1)

k, l = 0.95, 2
c = np.concatenate((s, a), axis = 1)
c = c[c[:, 0].argsort()]
b0 = c[-int(c.shape[0] * 0.2):, 1:]
b1 = np.random.rand(int(p * 0.4), m)
b2 = np.zeros_like(b1)
for i in range(b2.shape[0]):
    for j in range(b2.shape[1]):
        g = b0[np.random.randint(0, b0.shape[0]), j]
        if np.random.rand() > k:
            g *= np.random.rand() * l
        b2[i, j] = g

b = np.concatenate((b0, b1, b2))