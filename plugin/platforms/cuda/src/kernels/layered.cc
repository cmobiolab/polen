// POLEN: layered polarization (same formulas as AmoebaReferenceMultipoleForce).

#ifndef MLPOL_PAIR_THOLE
#define MLPOL_PAIR_THOLE
#ifdef MLPOL_CROSS_THOLE
#define PAIR_THOLE(t1, t2) ((signbit(t1) != signbit(t2)) ? (real) MLPOL_CROSS_THOLE : (real) min(fabs(t1), fabs(t2)))
#else
#define PAIR_THOLE(t1, t2) ((real) min(fabs(t1), fabs(t2)))
#endif
#endif

#ifdef USE_PERIODIC
#define PERIODIC_ARGS , real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ
#define WRAP(d) APPLY_PERIODIC_TO_DELTA(d)
#else
#define PERIODIC_ARGS
#define WRAP(d)
#endif

/** lambda3/r^3, 3 lambda5/r^5, 15 lambda7/r^7 (Thole), como getAndScaleInverseRs. */
DEVICE void layeredTholeRs(float2 dtI, float2 dtJ, real r, real* rr) {
    real rInv = RECIP(r), r2Inv = rInv*rInv;
    rr[0] = rInv*r2Inv;
    rr[1] = 3*rr[0]*r2Inv;
    rr[2] = 5*rr[1]*r2Inv;
    real damp = dtI.x*dtJ.x;
    if (damp != 0) {
        real ratio = r/damp;
        real u = -PAIR_THOLE(dtI.y, dtJ.y)*ratio*ratio*ratio;
        if (u > -50) {
            real e = EXP(u);
            rr[0] *= 1 - e;
            rr[1] *= 1 - (1 - u)*e;
            rr[2] *= 1 - (1 - u + 0.6f*u*u)*e;
        }
    }
}

/** Group weight s: smooth minimum of the distances from the reference atom to the central atoms and C2 switch between R_IN and R_OUT. */
KERNEL void computeGroupSwitch(GLOBAL const real4* RESTRICT posq, GLOBAL const int* RESTRICT groupRef,
        GLOBAL const int* RESTRICT centerAtoms, GLOBAL real* RESTRICT groupS, GLOBAL real* RESTRICT groupDS,
        GLOBAL real* RESTRICT softminW PERIODIC_ARGS) {
    for (int m = GLOBAL_ID; m < NUM_LAYER_GROUPS; m += GLOBAL_SIZE) {
        real3 r = trimTo3(posq[groupRef[m]]);
        real dmin = 1e30f;
        for (int a = 0; a < NUM_CENTERS; a++) {
            real3 delta = r - trimTo3(posq[centerAtoms[a]]);
            WRAP(delta)
            real dist = SQRT(dot(delta, delta));
            softminW[m*NUM_CENTERS+a] = dist;
            dmin = min(dmin, dist);
        }
        real sum = 0;
        for (int a = 0; a < NUM_CENTERS; a++)
            sum += EXP(-(softminW[m*NUM_CENTERS+a]-dmin)/LAYER_TAU);
        real d = dmin - LAYER_TAU*LOG(sum);
        for (int a = 0; a < NUM_CENTERS; a++)
            softminW[m*NUM_CENTERS+a] = EXP(-(softminW[m*NUM_CENTERS+a]-dmin)/LAYER_TAU)/sum;
        real s, ds;
        if (d <= LAYER_R_IN) {
            s = 1;
            ds = 0;
        }
        else if (d >= LAYER_R_OUT) {
            s = 0;
            ds = 0;
        }
        else {
            real width = LAYER_R_OUT-LAYER_R_IN;
            real x = (d-LAYER_R_IN)/width;
            ds = -30*x*x*(1-x)*(1-x)/width;
            s = 1 - x*x*x*(10 - 15*x + 6*x*x);
        }
        groupS[m] = s;
        groupDS[m] = ds;
    }
}

/** Per-atom weight and compact lists: active sites (s > 0) and sites in the switching zone (dS/dd != 0). */
KERNEL void computeAtomWeights(GLOBAL const int* RESTRICT groupOf, GLOBAL const real* RESTRICT groupS,
        GLOBAL const real* RESTRICT groupDS, GLOBAL real* RESTRICT atomWeight, GLOBAL int* RESTRICT activeList,
        GLOBAL int* RESTRICT transList, GLOBAL int* RESTRICT counters) {
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        int m = groupOf[i];
        real s = (m >= 0 ? groupS[m] : 1);
        atomWeight[i] = s;
        if (s > 0)
            activeList[ATOMIC_ADD(&counters[0], 1)] = i;
        if (m >= 0 && groupDS[m] != 0)
            transList[ATOMIC_ADD(&counters[1], 1)] = i;
    }
}

/** Induced–induced coupling field: direct sum among active sites, Thole, weight s_i s_j, no Ewald. */
KERNEL void layeredInducedField(GLOBAL const real4* RESTRICT posq, GLOBAL const float2* RESTRICT dampingAndThole,
        GLOBAL const int* RESTRICT activeList, GLOBAL const int* RESTRICT counters, GLOBAL const real* RESTRICT atomWeight,
        GLOBAL const real* RESTRICT inducedDipole, GLOBAL const real* RESTRICT inducedDipolePolar,
        GLOBAL mm_long* RESTRICT inducedField, GLOBAL mm_long* RESTRICT inducedFieldPolar PERIODIC_ARGS) {
    int numActive = counters[0];
    for (int t = GLOBAL_ID; t < numActive; t += GLOBAL_SIZE) {
        int i = activeList[t];
        real3 posI = trimTo3(posq[i]);
        float2 dtI = dampingAndThole[i];
        real sI = atomWeight[i];
        real3 fd = make_real3(0), fp = make_real3(0);
        for (int u = 0; u < numActive; u++) {
            int j = activeList[u];
            if (j == i)
                continue;
            real3 R = trimTo3(posq[j]) - posI;
            WRAP(R)
            real r = SQRT(dot(R, R));
            real rr[3];
            layeredTholeRs(dtI, dampingAndThole[j], r, rr);
            real w = sI*atomWeight[j];
            real rr3 = -rr[0]*w, rr5 = rr[1]*w;
            real3 mdJ = make_real3(inducedDipole[3*j], inducedDipole[3*j+1], inducedDipole[3*j+2]);
            real3 mpJ = make_real3(inducedDipolePolar[3*j], inducedDipolePolar[3*j+1], inducedDipolePolar[3*j+2]);
            fd += mdJ*rr3 + R*(rr5*dot(mdJ, R));
            fp += mpJ*rr3 + R*(rr5*dot(mpJ, R));
        }
        inducedField[i] = realToFixedPoint(fd.x);
        inducedField[i+PADDED_NUM_ATOMS] = realToFixedPoint(fd.y);
        inducedField[i+2*PADDED_NUM_ATOMS] = realToFixedPoint(fd.z);
        inducedFieldPolar[i] = realToFixedPoint(fp.x);
        inducedFieldPolar[i+PADDED_NUM_ATOMS] = realToFixedPoint(fp.y);
        inducedFieldPolar[i+2*PADDED_NUM_ATOMS] = realToFixedPoint(fp.z);
    }
}

/** Forces of the induced–induced coupling with the converged dipoles (addLayeredCouplingForces). */
KERNEL void layeredCouplingForces(GLOBAL const real4* RESTRICT posq, GLOBAL const float2* RESTRICT dampingAndThole,
        GLOBAL const int* RESTRICT activeList, GLOBAL const int* RESTRICT counters, GLOBAL const real* RESTRICT atomWeight,
        GLOBAL const real* RESTRICT inducedDipole, GLOBAL const real* RESTRICT inducedDipolePolar,
        GLOBAL mm_ulong* RESTRICT forceBuffers PERIODIC_ARGS) {
    int numActive = counters[0];
    for (int t = GLOBAL_ID; t < numActive; t += GLOBAL_SIZE) {
        int i = activeList[t];
        real3 posI = trimTo3(posq[i]);
        float2 dtI = dampingAndThole[i];
        real sI = atomWeight[i];
        real3 mdI = make_real3(inducedDipole[3*i], inducedDipole[3*i+1], inducedDipole[3*i+2]);
        real3 mpI = make_real3(inducedDipolePolar[3*i], inducedDipolePolar[3*i+1], inducedDipolePolar[3*i+2]);
        real3 force = make_real3(0);
        for (int u = 0; u < numActive; u++) {
            int j = activeList[u];
            if (j == i)
                continue;
            real3 R = trimTo3(posq[j]) - posI;
            WRAP(R)
            real r = SQRT(dot(R, R));
            real rr[3];
            layeredTholeRs(dtI, dampingAndThole[j], r, rr);
            real w = sI*atomWeight[j];
            real3 mdJ = make_real3(inducedDipole[3*j], inducedDipole[3*j+1], inducedDipole[3*j+2]);
            real3 mpJ = make_real3(inducedDipolePolar[3*j], inducedDipolePolar[3*j+1], inducedDipolePolar[3*j+2]);
            real aR = dot(mdI, R), bR = dot(mpJ, R), cR = dot(mdJ, R), dR = dot(mpI, R);
            real3 dF = R*(rr[1]*dot(mdI, mpJ) - rr[2]*aR*bR) + (mdI*bR + mpJ*aR)*rr[1]
                     + R*(rr[1]*dot(mdJ, mpI) - rr[2]*cR*dR) + (mdJ*dR + mpI*cR)*rr[1];
            force += dF*(-0.5f*w*ENERGY_SCALE_FACTOR);
        }
        forceBuffers[i] += (mm_ulong) realToFixedPoint(force.x);
        forceBuffers[i+PADDED_NUM_ATOMS] += (mm_ulong) realToFixedPoint(force.y);
        forceBuffers[i+2*PADDED_NUM_ATOMS] += (mm_ulong) realToFixedPoint(force.z);
    }
}

/** dE/ds_i of the sites in the switching zone: -1/2 (mu_d,i . F^p_i + mu_p,i . F^d_i), F^x_i = sum_j s_j T~_ij mu_x,j. */
KERNEL void layeredSwitchDerivatives(GLOBAL const real4* RESTRICT posq, GLOBAL const float2* RESTRICT dampingAndThole,
        GLOBAL const int* RESTRICT activeList, GLOBAL const int* RESTRICT transList, GLOBAL const int* RESTRICT counters,
        GLOBAL const real* RESTRICT atomWeight, GLOBAL const real* RESTRICT inducedDipole,
        GLOBAL const real* RESTRICT inducedDipolePolar, GLOBAL real* RESTRICT dEds PERIODIC_ARGS) {
    int numActive = counters[0], numTrans = counters[1];
    for (int t = GLOBAL_ID; t < numTrans; t += GLOBAL_SIZE) {
        int i = transList[t];
        real3 posI = trimTo3(posq[i]);
        float2 dtI = dampingAndThole[i];
        real3 fP = make_real3(0), fD = make_real3(0);
        for (int u = 0; u < numActive; u++) {
            int j = activeList[u];
            if (j == i)
                continue;
            real3 R = trimTo3(posq[j]) - posI;
            WRAP(R)
            real r = SQRT(dot(R, R));
            real rr[3];
            layeredTholeRs(dtI, dampingAndThole[j], r, rr);
            real rr3 = -rr[0], rr5 = rr[1], sJ = atomWeight[j];
            real3 mdJ = make_real3(inducedDipole[3*j], inducedDipole[3*j+1], inducedDipole[3*j+2]);
            real3 mpJ = make_real3(inducedDipolePolar[3*j], inducedDipolePolar[3*j+1], inducedDipolePolar[3*j+2]);
            fP += (mpJ*rr3 + R*(rr5*dot(mpJ, R)))*sJ;
            fD += (mdJ*rr3 + R*(rr5*dot(mdJ, R)))*sJ;
        }
        real3 mdI = make_real3(inducedDipole[3*i], inducedDipole[3*i+1], inducedDipole[3*i+2]);
        real3 mpI = make_real3(inducedDipolePolar[3*i], inducedDipolePolar[3*i+1], inducedDipolePolar[3*i+2]);
        dEds[i] = -0.5f*ENERGY_SCALE_FACTOR*(dot(mdI, fP) + dot(mpI, fD));
    }
}

/** Switching forces: dE/dd_m = (sum dE/ds_i) dS/dd, through the gradient of the smooth minimum (reference and central atoms). */
KERNEL void layeredSwitchForces(GLOBAL const real4* RESTRICT posq, GLOBAL const int* RESTRICT groupRef,
        GLOBAL const int* RESTRICT centerAtoms, GLOBAL const real* RESTRICT groupDS, GLOBAL const real* RESTRICT softminW,
        GLOBAL const int* RESTRICT groupAtomStart, GLOBAL const int* RESTRICT groupAtoms, GLOBAL const real* RESTRICT dEds,
        GLOBAL mm_ulong* RESTRICT forceBuffers PERIODIC_ARGS) {
    for (int m = GLOBAL_ID; m < NUM_LAYER_GROUPS; m += GLOBAL_SIZE) {
        if (groupDS[m] == 0)
            continue;
        real sum = 0;
        for (int k = groupAtomStart[m]; k < groupAtomStart[m+1]; k++)
            sum += dEds[groupAtoms[k]];
        real g = sum*groupDS[m];
        int ref = groupRef[m];
        real3 r = trimTo3(posq[ref]);
        real3 fRef = make_real3(0);
        for (int a = 0; a < NUM_CENTERS; a++) {
            int c = centerAtoms[a];
            real3 delta = r - trimTo3(posq[c]);
            WRAP(delta)
            real dist = SQRT(dot(delta, delta));
            real3 f = delta*(softminW[m*NUM_CENTERS+a]/dist*g);
            fRef -= f;
            ATOMIC_ADD(&forceBuffers[c], (mm_ulong) realToFixedPoint(f.x));
            ATOMIC_ADD(&forceBuffers[c+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(f.y));
            ATOMIC_ADD(&forceBuffers[c+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(f.z));
        }
        ATOMIC_ADD(&forceBuffers[ref], (mm_ulong) realToFixedPoint(fRef.x));
        ATOMIC_ADD(&forceBuffers[ref+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(fRef.y));
        ATOMIC_ADD(&forceBuffers[ref+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(fRef.z));
    }
}
