import numpy as np

def incremental_svd(ys, rank_data, tol=1e-15):
    U      = rank_data["U"]
    S      = rank_data["S"]
    Vh     = rank_data["V"].T
    ys     = np.array(ys)


    m = ys.shape[0]
    k = np.diag(S).shape[0] if S.size else 0

    d = U.T @ ys
    residual = ys - U @ d
    p = np.linalg.norm(residual)

    Q = np.block([
        [S, d.reshape(-1, 1)],
        [np.zeros((1, k)), np.array([[p]])]
    ])

    U_q, S_q, V_qh = np.linalg.svd(Q, full_matrices=False)

    if p < tol or k >= U.shape[1]:
        U = U @ U_q[:k, :] 
        S = np.diag(S_q)
        Vh = V_qh @ np.vstack([Vh, np.zeros((1, Vh.shape[1]))])
    else:
        j = residual / p
        U = np.column_stack([U, j]) @ U_q
        Vh = V_qh @ np.block([
            [Vh, np.zeros((k, 1))],
            [np.zeros((1, Vh.shape[1])), np.ones((1, 1))]
        ])
        S = np.diag(S_q)

    r = np.sum(np.diag(S) > tol)
    if r < len(np.diag(S)):
        U = U[:, :r]
        S = S[:r, :r]
        Vh = Vh[:r, :]

    # Orthogonalize outputs
    U, R_u = np.linalg.qr(U)
    S      = R_u @ S
    U_s, S_new, Vh_s = np.linalg.svd(S, full_matrices=False)
    U = U @ U_s
    S = np.diag(S_new)
    Vh = Vh_s @ Vh


    rank_data["U"] = U
    rank_data["S"] = S
    rank_data["V"] = Vh.T
#    print(Ushape)

    return rank_data

def initializeISVD(u1, W):
    S = np.sqrt(u1.T @ W @ u1)
    Q = u1 / S
    R = np.eye(1, 1)

    S *= np.eye(1, 1)
    Q = Q.reshape((Q.shape[0], 1))
    return Q, S, R

def updateISVD(Q, S, R, u_l, W, tol):
    d = Q.T @ W @ u_l
    if not np.shape(d):
        d *= np.eye(1, 1)
    e = u_l - Q @ d
    p = np.sqrt(e.T @ W @ e) * np.eye(1, 1)

    if p < tol:
        p = np.zeros((1, 1))
    else:
        e = e / p[0, 0].item()

    k = np.shape(S)[0] if np.shape(S) else 1
    Y = np.vstack((np.hstack((S, d)), np.hstack((np.zeros((1, k)), p))))
    Qy, Sy, Ry = np.linalg.svd(Y, full_matrices=True, compute_uv=True)
    Sy = np.diag(Sy)

    l = np.shape(R)[0]
    if p < tol:
        Q = Q @ Qy[:k, :k]
        S = Sy[:k, :k]
        R = np.vstack((np.hstack((R, np.zeros((l, 1)))), np.hstack(
            (np.zeros((1, k)), np.eye(1))))) @ Ry[:, :k]
    else:
        Q = np.hstack((Q, e)) @ Qy
        S = Sy
        R = np.vstack((np.hstack((R, np.zeros((l, 1)))),
                   np.hstack((np.zeros((1, k)), np.eye(1))))) @ Ry

    return Q, S, R

def modified_gram_schmidt(Q, W, tol):
    if np.abs(Q[:, -1].T @ W @ Q[:, 0]) > tol:
        k = Q.shape[1]
        for i in range(k):
            a = Q[:, i]
            for j in range(i):
                Q[:, i] = Q[:, i] - ((a.T @ W @ Q[:, j]) /
                                     (Q[:, j].T @ W @ Q[:, j])) * Q[:, j]
            norm = np.sqrt(Q[:, i].T @ W @ Q[:, i])
            Q[:, i] = Q[:, i] / norm
    return Q

def iSVD(U, W, tol=1e-15):
    m, n = np.shape(U)[0], np.shape(U)[1]
    u_0 = U[:, 0].reshape((m, 1))
    print(u_0.shape)
    Q, S, R = initializeISVD(u_0, W)
    for i in range(1, n):
        u_l = U[:, i].reshape((m, 1))
        Q, S, R = updateISVD(Q, S, R, u_l, W, tol)
        print(Q.shape)
        Q = modified_gram_schmidt(Q, W, tol)
    return Q, S, R

def main():
    rank_data = {
        "U": 0,
        "S": 0, 
        "V": 0,
    }
    W = np.eye(4000,4000)
    A = np.random.rand(4000, 4000) @ np.random.rand(4000, 50)
    print("wuh")
    U, _, _ = np.linalg.svd(A)
    for i in range(50):
        As = A[:,i].reshape(-1,1)
        if i == 0:
            norm = np.linalg.norm(As)
            rank_data["U"] = (As / norm).reshape(-1,1)
            rank_data["S"] = np.array([[norm]]).reshape(-1,1)
            rank_data["V"] = np.array([[1.0]]).reshape(-1,1)
            print(As.shape)
        else:
            rank_data = incremental_svd(As, rank_data)

    Q_2, S_2, R_2 = iSVD(A, W)

    A_approx = rank_data["U"] @ rank_data["S"] @ rank_data["V"].T
    A_approx_2 = Q_2 @ S_2 @ R_2
    error = np.linalg.norm(A-A_approx)/np.linalg.norm(A)
    print("Full space error: ", error)
    error = np.linalg.norm(A-A_approx_2)/np.linalg.norm(A)
    print("Full space error 2: ", error)
    U_approx = rank_data["U"]
    Ur = U[:,:6]
    Ur_approx = U_approx[:,:6]
    Ur_approx_2 = Q_2[:,:6]
    error = np.linalg.norm(Ur @ Ur.T-Ur_approx @ Ur_approx.T)/np.linalg.norm(Ur @ Ur.T)
    print("Reduced space error: ", error)
    error = np.linalg.norm(Ur @ Ur.T-Ur_approx_2 @ Ur_approx_2.T)/np.linalg.norm(Ur @ Ur.T)
    print("Reduced space error 2: ", error)

if __name__ == "__main__": 
    main()
