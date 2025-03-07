import numpy as np
import random

def combine(np1, xp, npc, xpc, rkf1, df):
    """Combine two clusters."""
    nits = 5000  # Reduced number of iterations
    pi = 4.0 * np.arctan(1.0)

    print(f"Combining clusters: np1={np1}, npc={npc}")

    if np1 == 1 and npc == 1:
        # Ensure xp has at least 2 columns
        if xp.shape[1] < 2:
            xp = np.zeros((3, 2))  # Resize xp to have 2 columns
        ct = 1.0 - 2.0 * random.random()
        phi = 2.0 * pi * random.random()
        st = np.sqrt((1.0 - ct) * (1.0 + ct))
        xp[:, 0] = np.array([st * np.cos(phi), st * np.sin(phi), ct])
        xp[:, 1] = -xp[:, 0]
        print("Single sphere case handled")
        return

    if npc == 1:
        print("Adding one sphere to the cluster")
        addone(np1, xp, rkf1, df)
        return

    rgc = npc ** (1.0 / df) * 2.0 * rkf1
    np3 = np1 + npc
    rg1 = np1 ** (1.0 / df) * 2.0 * rkf1
    rg3 = np3 ** (1.0 / df) * 2.0 * rkf1
    c = np.sqrt(np3 * (np3 * rg3**2 - np1 * rg1**2 - npc * rgc**2) / (npc * (np3 - npc)))

    # Initialize r12c to a large value
    r12c = float('inf')

    print(f"Starting combination loop with {nits} iterations")
    for it in range(nits):
        ctc = 1.0 - 2.0 * random.random()
        stc = np.sqrt((1.0 - ctc) * (1.0 + ctc))
        pc = 2.0 * pi * random.random()
        xc = np.array([c * stc * np.cos(pc), c * stc * np.sin(pc), c * ctc])
        xpc_new = xpc + xc[:, np.newaxis]

        ic1, ic2, r12 = finmin(0, np1, xp, npc, xpc_new)

        if r12 > 4.0:
            continue

        r1 = np.linalg.norm(xp[:, ic1] - xc)
        r2 = np.linalg.norm(xpc[:, ic2])
        if abs(r1 - r2) > 2.0:
            continue

        rc2 = ctos(xpc[:, ic2])
        beta = np.arccos(rc2[1])
        alpha = rc2[2]
        x1 = rotate(0, alpha, beta, 0.0, xp[:, ic1] - xc)
        gamma = np.arctan2(x1[1], x1[0])

        x1 = rotate(0, 0.0, 0.0, gamma, x1)
        rc1 = ctos(x1)
        ctc = (rc1[0]**2 + rc2[0]**2 - 4.0) / (2.0 * rc1[0] * rc2[0])
        if abs(ctc) > 1.0:
            continue
        tc = np.arccos(ctc)
        betac = tc - np.arccos(rc1[1])

        for i in range(npc):
            xpc[:, i] = rotate(0, alpha, beta, gamma, xpc[:, i])
            xpc[:, i] = rotate(0, 0.0, betac, 0.0, xpc[:, i])
            xpc[:, i] = rotate(1, alpha, beta, gamma, xpc[:, i])
            xpc_new[:, i] = xpc[:, i] + xc

        ic1, ic2, r12c = finmin(0, np1, xp, npc, xpc_new)

        if r12c >= 2.0:
            print(f"Valid configuration found at iteration {it}")
            break

    if r12c < 2.0:
        print("Clusters did not combine")
        return

    # Ensure xp has enough columns to store the new spheres
    if xp.shape[1] < np3:
        xp_new = np.zeros((3, np3))
        xp_new[:, :xp.shape[1]] = xp
        xp = xp_new

    # Assign xpc_new to xp only if np1 < np3
    if np1 < np3:
        xp[:, np1:np3] = xpc_new
    else:
        print("Warning: np1 >= np3, skipping assignment")

    xc = np.mean(xp[:, :np3], axis=1)
    xp[:, :np3] -= xc[:, np.newaxis]
    print("Clusters combined successfully")

def addone(nptot, xp, rkf1, df):
    """Add one sphere to the cluster."""
    itmax = 20000  # Maximum iterations for finding a valid position
    max_attempts = 1000  # Maximum attempts to increase rn
    attempt_count = 0  # Counter for attempts to increase rn

    print(f"Adding one sphere to the cluster with {nptot} spheres")

    rgn2 = np.sum(xp**2) / nptot
    rgn = np.sqrt(rgn2)
    rmax = np.sqrt(np.max(np.sum(xp**2, axis=0)))

    np3 = nptot + 1
    rg3 = np3 ** (1.0 / df) * 2.0 * rkf1
    rn2 = np3 * (np3 / nptot * rg3**2 - rgn2)
    rn = np.sqrt(rn2)

    print(f"Initial rn: {rn}, rmax: {rmax}")

    if rn - rmax > 2.0:
        rn = rmax + 1.8
        rn2 = rn**2
        rg32 = (rn2 / np3 + rgn2) * nptot / np3
        rg3 = np.sqrt(rg32)
        print(f"Adjusted rn: {rn}")

    while attempt_count < max_attempts:
        ijp = []
        rp = []
        for j in range(nptot):
            rj2 = np.sum(xp[:, j]**2)
            rj = np.sqrt(rj2)
            rjn = abs(rj - rn)
            if rjn <= 2.0:
                ijp.append(j)
                rp.append(rj)

        nj = len(ijp)
        print(f"Number of candidate spheres (nj): {nj}")

        if nj == 0:
            rn += 0.01
            rn2 = rn**2
            rg32 = (rn2 / np3 + rgn2) * nptot / np3
            rg3 = np.sqrt(rg32)
            attempt_count += 1
            print(f"No candidate spheres found, increasing rn to {rn}, attempt {attempt_count}/{max_attempts}")
            continue

        for ij in range(nj):
            j = ijp[ij]
            rj = rp[ij]
            rj2 = rj**2
            if rj + rn < 2.0:
                print(f"Sphere {j} skipped: rj + rn < 2.0")
                continue

            # Avoid division by zero
            if rj == 0.0:
                print(f"Sphere {j} skipped: rj is zero")
                continue  # Skip this sphere if rj is zero

            ctj = xp[2, j] / rj
            stj = np.sqrt((1.0 - ctj) * (1.0 + ctj))
            phij = np.arctan2(xp[1, j], xp[0, j])
            sphij = np.sin(phij)
            cphij = np.cos(phij)

            # Avoid division by zero
            if rn == 0.0 or rj == 0.0:
                print(f"Sphere {j} skipped: rn or rj is zero")
                continue  # Skip this sphere if rn or rj is zero

            ctp = (rn2 + rj2 - 4.0) / (2.0 * rn * rj)
            stp = np.sqrt((1.0 - ctp) * (1.0 + ctp))

            for it in range(itmax):
                phi = 2.0 * np.pi * random.random()
                zpp = rn * ctp
                xpp = rn * stp * np.cos(phi)
                ypp = rn * stp * np.sin(phi)
                z = zpp * ctj - xpp * stj
                x = (zpp * stj + xpp * ctj) * cphij - ypp * sphij
                y = (zpp * stj + xpp * ctj) * sphij + ypp * cphij

                icon = 0
                for i in range(nptot):
                    ri2 = (x - xp[0, i])**2 + (y - xp[1, i])**2 + (z - xp[2, i])**2
                    if ri2 < 3.999:
                        icon = 1
                        break

                if icon == 0:
                    # Ensure xp has enough columns to store the new sphere
                    if xp.shape[1] < np3:
                        xp_new = np.zeros((3, np3))
                        xp_new[:, :nptot] = xp
                        xp = xp_new
                    xp[:, np3 - 1] = np.array([x, y, z])
                    x0 = np.mean(xp[:, :np3], axis=1)
                    xp[:, :np3] -= x0[:, np.newaxis]
                    print(f"New sphere added successfully at iteration {it}")
                    return

        rn += 0.01
        rn2 = rn**2
        rg32 = (rn2 / np3 + rgn2) * nptot / np3
        rg3 = np.sqrt(rg32)
        attempt_count += 1
        print(f"No valid position found, increasing rn to {rn}, attempt {attempt_count}/{max_attempts}")

    # If we reach here, the maximum number of attempts has been reached
    print(f"Failed to add a new sphere after {max_attempts} attempts. Terminating.")
    return

def ctos(x):
    """Convert Cartesian coordinates to spherical coordinates."""
    r = np.linalg.norm(x)
    if r == 0.0:
        return np.array([0.0, 1.0, 0.0])
    else:
        return np.array([r, x[2] / r, np.arctan2(x[1], x[0])])

def rotate(idir, alpha, beta, gamma, x):
    """Rotate a point in 3D space."""
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    sb = np.sin(beta)
    cb = np.cos(beta)
    sg = np.sin(gamma)
    cg = np.cos(gamma)

    if idir == 0:
        xt = np.array([
            (ca * cb * cg - sa * sg) * x[0] + (cb * cg * sa + ca * sg) * x[1] - cg * sb * x[2],
            (-cg * sa - ca * cb * sg) * x[0] + (ca * cg - cb * sa * sg) * x[1] + sb * sg * x[2],
            ca * sb * x[0] + sa * sb * x[1] + cb * x[2]
        ])
    else:
        xt = np.array([
            (ca * cb * cg - sa * sg) * x[0] - (cb * sg * ca + sa * cg) * x[1] + ca * sb * x[2],
            (sg * ca + sa * cb * cg) * x[0] + (ca * cg - cb * sa * sg) * x[1] + sb * sa * x[2],
            -cg * sb * x[0] + sg * sb * x[1] + cb * x[2]
        ])

    return xt

def finmin(isame, np1, xp, np2, xpc):
    """Find the minimum distance between two sets of points."""
    rmin = 10000.0
    ic1 = 0
    ic2 = 0
    for i in range(np1):
        for j in range(np2):
            if isame == 0 or i != j:
                rij = np.linalg.norm(xp[:, i] - xpc[:, j])
                if rij < rmin:
                    rmin = rij
                    ic1 = i
                    ic2 = j
    return ic1, ic2, rmin

def ppclus(nptot, nsamp, rkf1, df, iccmod, xp, iseed0):
    """Generate a fractal cluster."""
    if iseed0 <= 0:
        random.seed()  # Seed with system time
    else:
        random.seed(iseed0)

    print(f"Generating fractal cluster with {nptot} particles")

    for irun in range(1000):
        xpt = np.zeros((3, nsamp))
        iadd = np.arange(nsamp)
        npc = np.ones(nsamp, dtype=int)
        nc = nsamp

        while nc > 1:
            i = 1
            j = 1
            while i == j:
                if iccmod == 0:
                    i = int(nc * random.random()) + 1
                else:
                    i = 1
                j = int(nc * random.random()) + 1

            ic = min(i, j)
            jc = max(i, j)
            iaddic = iadd[ic - 1]
            iaddjc = iadd[jc - 1]
            npic = npc[ic - 1]
            npjc = npc[jc - 1]

            xpc = xpt[:, iaddjc:iaddjc + npjc]

            for k in range(jc - 1, ic, -1):
                for nk in range(npc[k - 1], 0, -1):
                    i = nk + iadd[k - 1] - 1
                    j = i + npjc
                    xpt[:, j] = xpt[:, i]
                iadd[k - 1] += npjc

            combine(npic, xpt[:, iaddic:iaddic + npic], npjc, xpc, rkf1, df)
            npc[ic - 1] = npic + npjc

            if npc[ic - 1] == nptot:
                xp[:, :nptot] = xpt[:, iaddic:iaddic + nptot]
                print("Fractal cluster generation completed")
                return

            for j in range(jc - 1, nc - 1):
                jp = j + 1
                iadd[j] = iadd[jp]
                npc[j] = npc[jp]
            nc -= 1

    print("Process did not work. Try again.")

def main():
    # Example usage
    npart = 100
    nsamp = 2 * npart
    rk = 1.3
    df = 1.8
    iccmod = 0
    fout = "test.txt"

    rkf1 = (1.0 / rk) ** (1.0 / df) / 2.0
    xp = np.zeros((3, nsamp))
    iseed = 0

    ppclus(npart, nsamp, rkf1, df, iccmod, xp, iseed)

    # Save the output to a file
    with open(fout, "w") as f:
        for i in range(npart):
            f.write(f"1.0000 {xp[0, i]:10.4f} {xp[1, i]:10.4f} {xp[2, i]:10.4f}\n")

if __name__ == "__main__":
    main()