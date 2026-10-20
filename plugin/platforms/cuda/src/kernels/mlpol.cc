// POLEN: auxiliary kernels of the ML region.

/**
 * Inject the ML-atom parameters (lab frame, NoAxisType). mlParams per atom: q, mu (3),
 * Theta (xx, xy, xz, yy, yz), alpha, damping.
 */
KERNEL void injectMLParameters(GLOBAL real4* RESTRICT posq, GLOBAL float* RESTRICT localDipoles,
        GLOBAL float* RESTRICT localQuadrupoles, GLOBAL float* RESTRICT polarizability,
        GLOBAL float2* RESTRICT dampingAndThole, GLOBAL int4* RESTRICT multipoleParticles,
        GLOBAL const int* RESTRICT mlIndex, GLOBAL const real* RESTRICT mlParams, float mlThole) {
    for (int k = GLOBAL_ID; k < NUM_ML; k += GLOBAL_SIZE) {
        int i = mlIndex[k];
        GLOBAL const real* p = &mlParams[11*k];
        real4 pos = posq[i];
        pos.w = p[0];
        posq[i] = pos;
        for (int a = 0; a < 3; a++)
            localDipoles[3*i+a] = (float) p[1+a];
        for (int a = 0; a < 5; a++)
            localQuadrupoles[5*i+a] = (float) p[4+a];
        polarizability[i] = (float) p[9];
        dampingAndThole[i] = make_float2((float) p[10], -mlThole);
        multipoleParticles[i] = make_int4(-1, -1, -1, 5);
    }
}

// POLEN: pair Thole (same as the other kernels)
#ifndef MLPOL_PAIR_THOLE
#define MLPOL_PAIR_THOLE
#ifdef MLPOL_CROSS_THOLE
#define PAIR_THOLE(t1, t2) ((signbit(t1) != signbit(t2)) ? (real) MLPOL_CROSS_THOLE : (real) min(fabs(t1), fabs(t2)))
#else
#define PAIR_THOLE(t1, t2) ((real) min(fabs(t1), fabs(t2)))
#endif
#endif

#define NUM_DERIV 14

DEVICE void mlAddDipole(real3 x, real k3, real k5, real k7, real3 d, mixed* acc) {
    // contribution of a dipole x (kernels k3, k5, k7) to phi, -dE/dmu and G = dE/dTheta; R = -d
    real xd = dot(x, d);
    acc[0] += k3*xd;
    acc[1] += d.x*k5*xd - x.x*k3;
    acc[2] += d.y*k5*xd - x.y*k3;
    acc[3] += d.z*k5*xd - x.z*k3;
    real xv[3] = {x.x, x.y, x.z};
    real R[3] = {-d.x, -d.y, -d.z};
    for (int a = 0; a < 3; a++)
        for (int c = 0; c < 3; c++)
            acc[4+3*a+c] += xd*k7*R[a]*R[c] + 2*k5*xv[a]*R[c];
}

/**
 * Derivatives of the energy with respect to the ML-atom parameters (same formulas as
 * AmoebaReferenceMultipoleForce::mlDerivativesFromState and addMLReciprocalDerivatives). One group per ML atom.
 * Output per atom: phi, -dE/dmu (3), dE/dTheta (9, symmetric and traceless), dE/dalpha; in kJ/mol.
 */
KERNEL void computeMLDerivatives(GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT labDipole,
        GLOBAL const real* RESTRICT labQuadrupole, GLOBAL const real* RESTRICT inducedDipole,
        GLOBAL const real* RESTRICT inducedDipolePolar, GLOBAL const float2* RESTRICT dampingAndThole,
        GLOBAL const float* RESTRICT polarizability, GLOBAL const int* RESTRICT mlIndex,
        GLOBAL const int* RESTRICT specialStart, GLOBAL const int* RESTRICT specialAtom,
        GLOBAL const float4* RESTRICT specialScale, GLOBAL mixed* RESTRICT derivatives
#ifdef USE_EWALD
        , real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const real* RESTRICT phi, GLOBAL const real* RESTRICT phidp, real4 recipBoxVecX, real4 recipBoxVecY,
        real4 recipBoxVecZ, real plasmaTerm
#endif
#ifdef MLPOL_LAYERED
        , GLOBAL const real* RESTRICT atomWeight
#endif
        ) {
    LOCAL mixed buffer[NUM_DERIV*ML_THREADS];
    for (int k = GROUP_ID; k < NUM_ML; k += NUM_GROUPS) {
        int i = mlIndex[k];
        real4 posqI = posq[i];
        real3 posI = trimTo3(posqI);
        real3 muDI = make_real3(inducedDipole[3*i], inducedDipole[3*i+1], inducedDipole[3*i+2]);
        real3 muPI = make_real3(inducedDipolePolar[3*i], inducedDipolePolar[3*i+1], inducedDipolePolar[3*i+2]);
        real3 dipI = make_real3(labDipole[3*i], labDipole[3*i+1], labDipole[3*i+2]);
        real QI[9];
        QI[0] = labQuadrupole[5*i]; QI[1] = labQuadrupole[5*i+1]; QI[2] = labQuadrupole[5*i+2];
        QI[4] = labQuadrupole[5*i+3]; QI[5] = labQuadrupole[5*i+4]; QI[8] = -QI[0]-QI[4];
        QI[3] = QI[1]; QI[6] = QI[2]; QI[7] = QI[5];
        float2 dtI = dampingAndThole[i];
        real alphaI = polarizability[i];
        int first = specialStart[k], last = specialStart[k+1];
        mixed acc[NUM_DERIV];
        for (int n = 0; n < NUM_DERIV; n++)
            acc[n] = 0;
        mixed gradAlpha = 0;
        for (int j = LOCAL_ID; j < NUM_ATOMS; j += LOCAL_SIZE) {
            if (j == i)
                continue;
            real4 posqJ = posq[j];
            real3 d = posI - trimTo3(posqJ);
#ifdef USE_EWALD
            APPLY_PERIODIC_TO_DELTA(d)
#endif
            real r2 = dot(d, d);
            // scaling factors: 1, except for the pairs in the (sorted) list of this ML atom
            float4 sc = make_float4(1, 1, 1, 1);
            int lo = first, hi = last-1;
            while (lo <= hi) {
                int mid = (lo+hi)/2;
                int a = specialAtom[mid];
                if (a == j) {
                    sc = specialScale[mid];
                    break;
                }
                if (a < j)
                    lo = mid+1;
                else
                    hi = mid-1;
            }
            real mS = sc.x, dS = sc.y, pS = sc.z, uS = sc.w;
#ifdef USE_EWALD
            // POLEN: a masked pair enters even beyond the cutoff (b_n - (1 - s) B_n + reciprocal = s B_n)
            if (r2 > CUTOFF_SQUARED && mS == 1 && dS == 1 && pS == 1 && uS == 1)
                continue;
#endif
            real r = SQRT(r2);
            real rInv = RECIP(r);
            real B[5], b[5], kp[5];
            B[0] = rInv;
            for (int n = 1; n < 5; n++)
                B[n] = (2*n-1)*B[n-1]*rInv*rInv;
#ifdef USE_EWALD
            real ralpha = EWALD_ALPHA*r;
            real exp2a = EXP(-ralpha*ralpha);
            real alsq2 = 2*EWALD_ALPHA*EWALD_ALPHA;
            real alsq2n = RECIP(SQRT_PI*EWALD_ALPHA);
            b[0] = erfc(ralpha)*rInv;
            for (int n = 1; n < 5; n++) {
                alsq2n *= alsq2;
                b[n] = ((2*n-1)*b[n-1] + alsq2n*exp2a)*rInv*rInv;
            }
#else
            for (int n = 0; n < 5; n++)
                b[n] = B[n];
#endif
            for (int n = 0; n < 5; n++)
                kp[n] = b[n] - (1-mS)*B[n];
            real qJ = posqJ.w;
            real3 dipJ = make_real3(labDipole[3*j], labDipole[3*j+1], labDipole[3*j+2]);
            real QJ[9];
            QJ[0] = labQuadrupole[5*j]; QJ[1] = labQuadrupole[5*j+1]; QJ[2] = labQuadrupole[5*j+2];
            QJ[4] = labQuadrupole[5*j+3]; QJ[5] = labQuadrupole[5*j+4]; QJ[8] = -QJ[0]-QJ[4];
            QJ[3] = QJ[1]; QJ[6] = QJ[2]; QJ[7] = QJ[5];
            real3 Qjd = make_real3(QJ[0]*d.x+QJ[1]*d.y+QJ[2]*d.z, QJ[3]*d.x+QJ[4]*d.y+QJ[5]*d.z, QJ[6]*d.x+QJ[7]*d.y+QJ[8]*d.z);
            real dQd = dot(Qjd, d);
            // permanent (m scaling)
            acc[0] += qJ*kp[0] + dQd*kp[2];
            acc[1] += d.x*(qJ*kp[1] + dQd*kp[3]) - Qjd.x*2*kp[2];
            acc[2] += d.y*(qJ*kp[1] + dQd*kp[3]) - Qjd.y*2*kp[2];
            acc[3] += d.z*(qJ*kp[1] + dQd*kp[3]) - Qjd.z*2*kp[2];
            {
                real R[3] = {-d.x, -d.y, -d.z};
                real QR[3] = {-Qjd.x, -Qjd.y, -Qjd.z};
                for (int a = 0; a < 3; a++)
                    for (int c = 0; c < 3; c++)
                        acc[4+3*a+c] += R[a]*R[c]*(qJ*kp[2] + dQd*kp[4]) + 2*QJ[3*a+c]*kp[2] - 4*R[a]*QR[c]*kp[3];
            }
            mlAddDipole(dipJ, kp[1], kp[2], kp[3], d, acc);
            // induced: Thole
            float2 dtJ = dampingAndThole[j];
            real pairA = PAIR_THOLE(dtI.y, dtJ.y);
            real damp = dtI.x*dtJ.x;
            real s3 = 1, s5 = 1, s7 = 1, rho = 0;
            if (damp != 0) {
                real ratio = r/damp;
                rho = pairA*ratio*ratio*ratio;
                if (rho < 50) {
                    real e = EXP(-rho);
                    s3 = 1 - e;
                    s5 = 1 - (1 + rho)*e;
                    s7 = 1 - (1 + rho + 0.6f*rho*rho)*e;
                }
            }
            real3 muDJ = make_real3(inducedDipole[3*j], inducedDipole[3*j+1], inducedDipole[3*j+2]);
            real3 muPJ = make_real3(inducedDipolePolar[3*j], inducedDipolePolar[3*j+1], inducedDipolePolar[3*j+2]);
            mlAddDipole((muPJ*dS + muDJ*pS)*0.5f, B[1]*s3, B[2]*s5, B[3]*s7, d, acc);
#ifdef USE_EWALD
            mlAddDipole((muPJ + muDJ)*0.5f, b[1]-B[1], b[2]-B[2], b[3]-B[3], d, acc);
#endif
            // dE/dalpha_i
            if (alphaI > 0 && damp != 0 && rho < 50) {
                real e = EXP(-rho);
                real g3 = rho*e, g5 = rho*rho*e, g7 = (0.6f*rho*rho*rho - 0.2f*rho*rho)*e;
                real rr3g = g3*B[1], rr5g = g5*B[2], rr7g = g7*B[3];
                real mud = dot(dipJ, d);
                real3 Gij = d*(qJ*rr3g + mud*rr5g + dQd*rr7g) - dipJ*rr3g - Qjd*(2*rr5g);
                real3 dji = d*(-1);
                real3 Qid = make_real3(QI[0]*dji.x+QI[1]*dji.y+QI[2]*dji.z, QI[3]*dji.x+QI[4]*dji.y+QI[5]*dji.z, QI[6]*dji.x+QI[7]*dji.y+QI[8]*dji.z);
                real3 Gji = dji*(posqI.w*rr3g + dot(dipI, dji)*rr5g + dot(Qid, dji)*rr7g) - dipI*rr3g - Qid*(2*rr5g);
                real asTarget = dS*dot(Gij, muPI) + pS*dot(muDI, Gij);
                real asSource = dS*dot(Gji, muPJ) + pS*dot(muDJ, Gji);
                real g5r5 = g5*B[2]/3, g3r3 = g3*B[1];
#ifdef MLPOL_LAYERED
                uS *= atomWeight[i]*atomWeight[j];      // coupling weighted by s_i s_j
#endif
                real tensor = uS*(3*g5r5*dot(muDI, d)*dot(muPJ, d) - g3r3*dot(muDI, muPJ))
                            + uS*(3*g5r5*dot(muDJ, d)*dot(muPI, d) - g3r3*dot(muDJ, muPI));
                gradAlpha += (asTarget + asSource + tensor)/(4*alphaI);
            }
        }
        // reduction within the group
        for (int n = 0; n < NUM_DERIV-1; n++)
            buffer[n*ML_THREADS+LOCAL_ID] = acc[n];
        buffer[(NUM_DERIV-1)*ML_THREADS+LOCAL_ID] = gradAlpha;
        SYNC_THREADS;
        for (int step = ML_THREADS/2; step > 0; step /= 2) {
            if (LOCAL_ID < step)
                for (int n = 0; n < NUM_DERIV; n++)
                    buffer[n*ML_THREADS+LOCAL_ID] += buffer[n*ML_THREADS+LOCAL_ID+step];
            SYNC_THREADS;
        }
        if (LOCAL_ID == 0) {
            mixed out[NUM_DERIV];
            for (int n = 0; n < NUM_DERIV; n++)
                out[n] = buffer[n*ML_THREADS];
            if (alphaI > 0)
                out[13] += -dot(muDI, muPI)/(2*alphaI*alphaI);
#ifdef USE_EWALD
            // reciprocal: (K M)_i + 1/2 potential of the induced dipoles, in Cartesian coordinates (as transformPotentialToCartesianCoordinates)
            real am[3][3];
            am[0][0] = GRID_SIZE_X*recipBoxVecX.x; am[1][0] = GRID_SIZE_X*recipBoxVecY.x; am[2][0] = GRID_SIZE_X*recipBoxVecZ.x;
            am[0][1] = GRID_SIZE_Y*recipBoxVecX.y; am[1][1] = GRID_SIZE_Y*recipBoxVecY.y; am[2][1] = GRID_SIZE_Y*recipBoxVecZ.y;
            am[0][2] = GRID_SIZE_Z*recipBoxVecX.z; am[1][2] = GRID_SIZE_Z*recipBoxVecY.z; am[2][2] = GRID_SIZE_Z*recipBoxVecZ.z;
            int index1[] = {0, 1, 2, 0, 0, 1};
            int index2[] = {0, 1, 2, 1, 2, 2};
            real bm[6][6];
            for (int a = 0; a < 6; a++)
                for (int c = 0; c < 6; c++) {
                    bm[a][c] = am[index1[a]][index1[c]]*am[index2[a]][index2[c]];
                    if (index1[c] != index2[c])
                        bm[a][c] += (a < 3 ? bm[a][c] : am[index1[a]][index2[c]]*am[index2[a]][index1[c]]);
                }
            real c[10];
            for (int n = 0; n < 10; n++)
                c[n] = 0;
            for (int src = 0; src < 2; src++) {
                GLOBAL const real* f = (src == 0 ? phi : phidp);
                real w = (src == 0 ? 1 : 0.5f);
                c[0] += w*f[i];
                for (int a = 0; a < 3; a++)
                    c[1+a] += w*(am[a][0]*f[i+NUM_ATOMS] + am[a][1]*f[i+NUM_ATOMS*2] + am[a][2]*f[i+NUM_ATOMS*3]);
                for (int a = 0; a < 6; a++)
                    for (int m = 0; m < 6; m++)
                        c[4+a] += w*bm[a][m]*f[i+NUM_ATOMS*(4+m)];
            }
            const int second[9] = {4, 7, 8, 7, 5, 9, 8, 9, 6};
            out[0] += c[0];
            for (int a = 0; a < 3; a++)
                out[1+a] -= c[1+a];
            for (int n = 0; n < 9; n++)
                out[4+n] += c[second[n]];
            // autotermos
            real fac = EWALD_ALPHA/SQRT_PI, a2 = EWALD_ALPHA*EWALD_ALPHA;
            real c1 = -2*fac*a2/3;
            out[0] += -2*fac*posqI.w + plasmaTerm;
            real3 self = dipI*(2*c1) + (muDI + muPI)*c1;
            out[1] -= self.x;
            out[2] -= self.y;
            out[3] -= self.z;
            real gq[9] = {6*(QI[0]-QI[4]), 12*QI[1], 12*QI[2], 12*QI[1], -6*(QI[0]-QI[4]), 12*QI[5], 12*QI[2], 12*QI[5], 18*QI[8]};
            for (int n = 0; n < 9; n++)
                out[4+n] -= fac*(4.0f/15.0f)*a2*a2*gq[n];
#endif
            // symmetric and traceless; energy units
            mixed tr = out[4] + out[8] + out[12];
            GLOBAL mixed* o = &derivatives[NUM_DERIV*k];
            o[0] = ENERGY_SCALE_FACTOR*out[0];
            for (int a = 0; a < 3; a++)
                o[1+a] = ENERGY_SCALE_FACTOR*out[1+a];
            for (int a = 0; a < 3; a++)
                for (int c2 = 0; c2 < 3; c2++)
                    o[4+3*a+c2] = ENERGY_SCALE_FACTOR*(0.5f*(out[4+3*a+c2] + out[4+3*c2+a]) - (a == c2 ? tr/3 : 0));
            o[13] = ENERGY_SCALE_FACTOR*out[13];
        }
        SYNC_THREADS;
    }
}

KERNEL void gatherMLPositions(GLOBAL const real4* RESTRICT posq, GLOBAL const int* RESTRICT mlIndex, GLOBAL real4* RESTRICT mlPos) {
    for (int k = GLOBAL_ID; k < NUM_ML; k += GLOBAL_SIZE)
        mlPos[k] = posq[mlIndex[k]];
}

KERNEL void addMLForces(GLOBAL mm_ulong* RESTRICT forceBuffers, GLOBAL const int* RESTRICT mlIndex, GLOBAL const real* RESTRICT mlForce) {
    for (int k = GLOBAL_ID; k < NUM_ML; k += GLOBAL_SIZE) {
        int i = mlIndex[k];
        forceBuffers[i] += (mm_ulong) realToFixedPoint(mlForce[3*k]);
        forceBuffers[i+PADDED_NUM_ATOMS] += (mm_ulong) realToFixedPoint(mlForce[3*k+1]);
        forceBuffers[i+2*PADDED_NUM_ATOMS] += (mm_ulong) realToFixedPoint(mlForce[3*k+2]);
    }
}

/**
 * Charge penetration: Slater valence of the ML atoms (N, s) × MM permanent multipoles (lab frame).
 * f(r) = N (1/r + 1/(2s)) e^{-r/s}; with b_0 = f/N and b_{k+1} = -(1/r) d b_k/dr, R = r_j - r_A:
 *   E = N [q b0 - (mu.R) b1 + (R.Theta.R) b2]   (traceless Theta)
 *   F_j = -N [-q R b1 + (mu.R) R b2 - mu b1 - (R.Theta.R) R b3 + 2 Theta.R b2],  F_A = -F_j
 *   tau_j = N [b1 mu x R + 2 b2 R x Theta.R]
 * Same energy as AmoebaReferenceMultipoleForce::addChargePenetration. One group per ML atom.
 */
KERNEL void computePenetration(GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT labDipole,
        GLOBAL const real* RESTRICT labQuadrupole, GLOBAL const float2* RESTRICT dampingAndThole,
        GLOBAL const int* RESTRICT mlIndex, GLOBAL const real* RESTRICT penParams,
        GLOBAL mm_ulong* RESTRICT forceBuffers, GLOBAL mm_ulong* RESTRICT torqueBuffers,
        GLOBAL mixed* RESTRICT energyBuffer, GLOBAL mixed* RESTRICT penDerivs
#ifdef USE_EWALD
        , real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ
#endif
        ) {
    LOCAL mixed buffer[2*ML_THREADS];
    mixed energy = 0;
    for (int k = GROUP_ID; k < NUM_ML; k += NUM_GROUPS) {
        int A = mlIndex[k];
        real3 posA = trimTo3(posq[A]);
        real N = penParams[2*k], s = penParams[2*k+1];
        real s2 = s*s, s3 = s2*s;
        mixed dN = 0, dS = 0;
        real3 forceA = make_real3(0);
        for (int j = LOCAL_ID; j < NUM_ATOMS; j += LOCAL_SIZE) {
            if (signbit(dampingAndThole[j].y))
                continue;                                   // ML atom
            real4 posqJ = posq[j];
            real3 R = trimTo3(posqJ) - posA;
#ifdef USE_EWALD
            APPLY_PERIODIC_TO_DELTA(R)
#endif
            real r2 = dot(R, R);
#ifdef USE_EWALD
            if (r2 > CUTOFF_SQUARED)
                continue;
#endif
            real r = SQRT(r2);
            if (r > 40*s)
                continue;                                   // e^{-r/s} negligible
            real e = EXP(-r/s);
            real rInv = RECIP(r), r3 = r2*r;
            real b0 = (0.5f*r + s)*rInv/s*e;
            real b1 = (0.5f*r2 + r*s + s2)/(r3*s2)*e;
            real b2 = (r3 + 3*r2*s + 6*r*s2 + 6*s3)/(2*r3*r2*s3)*e;
            real b3 = (r2*r2 + 5*r3*s + 15*r2*s2 + 30*r*s3 + 30*s2*s2)/(2*r3*r2*r2*s2*s2)*e;
            real db0 = (r + s)/(2*s3)*e, db1 = e/(2*s2*s2), db2 = e/(2*r*s3*s2);
            real q = posqJ.w;
            real3 mu = make_real3(labDipole[3*j], labDipole[3*j+1], labDipole[3*j+2]);
            real qxx = labQuadrupole[5*j], qxy = labQuadrupole[5*j+1], qxz = labQuadrupole[5*j+2];
            real qyy = labQuadrupole[5*j+3], qyz = labQuadrupole[5*j+4], qzz = -qxx-qyy;
            real3 w = make_real3(qxx*R.x + qxy*R.y + qxz*R.z, qxy*R.x + qyy*R.y + qyz*R.z, qxz*R.x + qyz*R.y + qzz*R.z);
            real muR = dot(mu, R), RQR = dot(R, w);
            real u = q*b0 - muR*b1 + RQR*b2;
            energy += ENERGY_SCALE_FACTOR*N*u;
            dN += u;
            dS += N*(q*db0 - muR*db1 + RQR*db2);
            real3 fJ = (R*(-q*b1) + R*(muR*b2) - mu*b1 - R*(RQR*b3) + w*(2*b2))*(-N*ENERGY_SCALE_FACTOR);
            real3 tJ = (cross(mu, R)*b1 + cross(R, w)*(2*b2))*(N*ENERGY_SCALE_FACTOR);
            forceA -= fJ;
            ATOMIC_ADD(&forceBuffers[j], (mm_ulong) realToFixedPoint(fJ.x));
            ATOMIC_ADD(&forceBuffers[j+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(fJ.y));
            ATOMIC_ADD(&forceBuffers[j+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(fJ.z));
            ATOMIC_ADD(&torqueBuffers[j], (mm_ulong) realToFixedPoint(tJ.x));
            ATOMIC_ADD(&torqueBuffers[j+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(tJ.y));
            ATOMIC_ADD(&torqueBuffers[j+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(tJ.z));
        }
        ATOMIC_ADD(&forceBuffers[A], (mm_ulong) realToFixedPoint(forceA.x));
        ATOMIC_ADD(&forceBuffers[A+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceA.y));
        ATOMIC_ADD(&forceBuffers[A+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceA.z));
        buffer[LOCAL_ID] = dN;
        buffer[ML_THREADS+LOCAL_ID] = dS;
        SYNC_THREADS;
        for (int step = ML_THREADS/2; step > 0; step /= 2) {
            if (LOCAL_ID < step) {
                buffer[LOCAL_ID] += buffer[LOCAL_ID+step];
                buffer[ML_THREADS+LOCAL_ID] += buffer[ML_THREADS+LOCAL_ID+step];
            }
            SYNC_THREADS;
        }
        if (LOCAL_ID == 0) {
            penDerivs[2*k] = ENERGY_SCALE_FACTOR*buffer[0];
            penDerivs[2*k+1] = ENERGY_SCALE_FACTOR*buffer[ML_THREADS];
        }
        SYNC_THREADS;
    }
    energyBuffer[GLOBAL_ID] += energy;
}
