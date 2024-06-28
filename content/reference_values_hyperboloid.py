from ngsolve import *
from ngsolve.meshes import Make1DMesh


def ComputeRefValue(n=200, order=2, t=0.1, ReissnerMindlin=True):
    mesh = Make1DMesh(n)

    E = 2.85e4
    nu = 0.3
    k = 5 / 6
    p0 = 1e4 * t**3

    if ReissnerMindlin:
        fes = (
            H1(mesh, order=order, dirichlet="left")
            * H1(mesh, order=order)
            * H1(mesh, order=order)
            * H1(mesh, order=order, dirichlet="left")
            * H1(mesh, order=order)
        )
        (u, v, w, alpha, beta), (du, dv, dw, dalpha, dbeta) = fes.TnT()
    else:  # Kirchhoff-Love
        fes = (
            H1(mesh, order=order, dirichlet="left")
            * H1(mesh, order=order)
            * H1(mesh, order=order)
            * H1(mesh, order=order, dirichlet="left")
            * H1(mesh, order=order)
            * L2(mesh, order=order - 1)
            * L2(mesh, order=order - 1)
        )
        (u, v, w, alpha, beta, xi1, xi2), (du, dv, dw, dalpha, dbeta, dxi1, dxi2) = (
            fes.TnT()
        )
        xi, dxi = CF((xi1, xi2)), CF((dxi1, dxi2))

    A = sqrt(1 + x**2)
    B = sqrt(1 + 2 * x**2)

    cf_gamma11 = Grad(u) - x / (A**2 * B**2) * u + 1 / (A**2 * B) * w
    cf_gamma22 = 2 * v + A**2 * x / B**2 * u - A**2 / B * w
    cf_gamma12 = 1 / 2 * (Grad(v) - 2 * u - 2 * x / A**2 * v)

    cf_d_gamma11 = Grad(du) - x / (A**2 * B**2) * du + 1 / (A**2 * B) * dw
    cf_d_gamma22 = 2 * dv + A**2 * x / B**2 * du - A**2 / B * dw
    cf_d_gamma12 = 1 / 2 * (Grad(dv) - 2 * du - 2 * x / A**2 * dv)

    cf_zeta1 = 1 / 2 * (alpha + Grad(w) - 1 / B**3 * u)
    cf_zeta2 = 1 / 2 * (beta - 2 * w + 1 / B * v)

    cf_d_zeta1 = 1 / 2 * (dalpha + Grad(dw) - 1 / B**3 * du)
    cf_d_zeta2 = 1 / 2 * (dbeta - 2 * dw + 1 / B * dv)

    cf_chi11 = (
        Grad(alpha)
        - x / (A**2 * B**2) * alpha
        + 1 / B**3 * (Grad(u) - x / (A**2 * B**2) * u)
        + 1 / (A**2 * B**4) * w
    )
    cf_chi22 = (
        2 * beta
        + x * A**2 / B**2 * alpha
        - 1 / B * (2 * v + x * A**2 / B**2 * u)
        + A**2 / B**2 * w
    )
    cf_chi12 = (
        1
        / 2
        * (
            -2 * alpha
            - 2 * x / A**2 * beta
            + Grad(beta)
            - 1 / B * Grad(v)
            - 2 / B**3 * u
            + 2 * x**3 * v / (A**2 * B**3)
        )
    )

    cf_d_chi11 = (
        Grad(dalpha)
        - x / (A**2 * B**2) * dalpha
        + 1 / B**3 * (Grad(du) - x / (A**2 * B**2) * du)
        + 1 / (A**2 * B**4) * dw
    )
    cf_d_chi22 = (
        2 * dbeta
        + x * A**2 / B**2 * dalpha
        - 1 / B * (2 * dv + x * A**2 / B**2 * du)
        + A**2 / B**2 * dw
    )
    cf_d_chi12 = (
        1
        / 2
        * (
            -2 * dalpha
            - 2 * x / A**2 * dbeta
            + Grad(dbeta)
            - 1 / B * Grad(dv)
            - 2 / B**3 * du
            + 2 * x**3 * dv / (A**2 * B**3)
        )
    )

    zeta = CF((cf_zeta1, cf_zeta2))
    gamma = CF((cf_gamma11, cf_gamma12, cf_gamma12, cf_gamma22), dims=(2, 2))
    chi = CF((cf_chi11, cf_chi12, cf_chi12, cf_chi22), dims=(2, 2))

    dzeta = CF((cf_d_zeta1, cf_d_zeta2))
    dgamma = CF((cf_d_gamma11, cf_d_gamma12, cf_d_gamma12, cf_d_gamma22), dims=(2, 2))
    dchi = CF((cf_d_chi11, cf_d_chi12, cf_d_chi12, cf_d_chi22), dims=(2, 2))

    a_inv = CF((A**2 / B**2, 0, 0, 1 / A**2), dims=(2, 2))
    vol = B

    shear_term = t * k * 2 * E / (1 + nu) * ((a_inv * zeta) * dzeta)
    membrane_term = CF(0)
    bending_term = CF(0)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    membrane_term += (
                        (
                            a_inv[i, k] * a_inv[j, l]
                            + a_inv[i, l] * a_inv[j, k]
                            + 2 * nu * a_inv[i, j] * a_inv[k, l] / (1 - nu)
                        )
                        * gamma[i, j]
                        * dgamma[k, l]
                    )
                    bending_term += (
                        (
                            a_inv[i, k] * a_inv[j, l]
                            + a_inv[i, l] * a_inv[j, k]
                            + 2 * nu * a_inv[i, j] * a_inv[k, l] / (1 - nu)
                        )
                        * chi[i, j]
                        * dchi[k, l]
                    )
    membrane_term *= t * E / (2 * (1 + nu))
    bending_term *= t**3 / 12 * E / (2 * (1 + nu))

    force = pi * vol * p0 * dw

    a = BilinearForm(fes)
    if ReissnerMindlin:
        a += (
            -pi
            * vol
            * (membrane_term + bending_term + shear_term)
            * dx(bonus_intorder=order)
        )
    else:
        a += (
            -pi
            * vol
            * (membrane_term + bending_term + a_inv * zeta * dxi + a_inv * dzeta * xi)
            * dx(bonus_intorder=order)
        )
    a.Assemble()

    f = LinearForm(fes)
    f += force * dx
    f.Assemble()

    gfsol = GridFunction(fes)

    inv = a.mat.Inverse(fes.FreeDofs(), inverse="umfpack")
    gfsol.vec.data = solvers.PreconditionedRichardson(
        a, f.vec, inv, maxit=3, tol=1e-13, printing=False
    )

    if ReissnerMindlin:
        gfu, gfv, gfw, gfalpha, gfbeta = gfsol.components
    else:
        gfu, gfv, gfw, gfalpha, gfbeta, _, _ = gfsol.components

    return gfw(mesh(0)), gfu, gfv, gfw, gfalpha, gfbeta


def Get3DSolution(n=200, order=2, t=0.1, ReissnerMindlin=True):
    _, gfu, gfv, gfw, _, _ = ComputeRefValue(n, order, t, ReissnerMindlin)
    A = sqrt(1 + x**2)
    B = sqrt(1 + 2 * x**2)
    A_sqr = 1 + x**2
    B_sqr = 1 + 2 * x**2

    theta = atan2(z, y)

    a1_inv = 1 / B_sqr * CF((A_sqr, A * x * cos(theta), A * x * sin(theta)))
    a2_inv = 1 / A * CF((0, -sin(theta), cos(theta)))
    a3 = B * Cross(a1_inv, a2_inv)

    # reconstruct displacement field from 1D solution
    displacement = (
        gfu * cos(2 * theta) * a1_inv
        + gfv * sin(2 * theta) * a2_inv
        + gfw * cos(2 * theta) * a3
    )

    return displacement
