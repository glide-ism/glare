// Depth-averaged snowpack enthalpy model (forward).
//
// Per-pixel water-year integration of an active surface snow reservoir carrying
// mass M [kg m-2] and enthalpy E [J m-2] (relative to ice at 0 C).  This is a
// GPU port of the toy scalar model in enthalpy_sketch_revised.py / the reference
// `step` in glare/enthalpy.py: same state convention, same implicit backward-Euler
// cold branch, same melt / runoff / ice-melt regimes.
//
// It is a *companion* to compute_smb (the PDD/radiation core) -- a physically
// explicit thermodynamic alternative for the snowpack, not a replacement.  One
// thread per pixel scans the water year (start_month -> +nt) once from a cold,
// snow-free surface (M = E = 0), mirroring compute_smb's single-water-year sweep.
// State persists month to month (no annual reset): the seasonal pack builds and
// ablates over the cycle.  Outputs are written in calendar-month order (idx uses
// the calendar index m), exactly as compute_smb does.
//
// Each month is integrated in n_sub explicit sub-steps (see the sub-loop) driven by
// the monthly-mean temperature plus a per-sub-step deviation from the spatially-uniform
// temp_dev vector (nt, n_sub).  This resolves the effect of sub-monthly temperature
// variance on the stateful pack; n_sub == 1 with zero deviations is the plain monthly
// model.  The randomness that fills temp_dev is generated in Python, never here.
//
// Retained-state invariant, enforced every step: M >= 0 and E <= 0.  A positive
// provisional enthalpy is meltwater -> converted to runoff (and, on a glacier
// surface, ice melt) at 0 C; it is never stored.
//
// Units: forcing precip arrives in m a-1 water-equivalent (glare convention) and
// is converted to SI mass with rho_w.  All state/flux outputs are SI (kg m-2,
// J m-2) and match the scalar reference to floating-point tolerance.  Burial of
// the active reservoir into ice (M_active_max in the reference Parameters) is not
// implemented in v1, matching the reference `step`.

__device__ __forceinline__ float enth_rain_fraction(float T_air, float width) {
    // Smooth rain/snow partition; -> 1 for warm air, -> 0 for cold.
    return 1.0f / (1.0f + expf(-T_air / width));
}

// Specific enthalpy of precipitation [J kg-1], referenced to ice at 0 C.  Solid
// precip is capped at <= 0 C and liquid at >= 0 C so the smoothing interval never
// implies warm ice or supercooled rain -- identical to the reference.
__device__ __forceinline__ float enth_precip_enthalpy(
        float T_air, float f_rain, float L_f, float c_i, float c_w) {
    float T_solid = fminf(T_air, 0.0f);
    float T_liquid = fmaxf(T_air, 0.0f);
    return (1.0f - f_rain) * c_i * T_solid + f_rain * (L_f + c_w * T_liquid);
}

extern "C" __global__ void compute_enthalpy(
    // The six diagnostic outputs (everything but smb_out) are deliberately
    // NOT __restrict__: with EnthalpyGrid(materialize_state=False) they all
    // alias one shared scratch cube. They are write-only, so aliased stores
    // are well-defined (same-thread program order; the final value is
    // garbage by contract). smb_out always has its own buffer.
    float* __restrict__ smb_out,        // (nt,ny,nx) surface mass balance rate [m a-1 w.e.]
    float* M_out,          // (nt,ny,nx) end-of-month active mass  [kg m-2]
    float* E_out,          // (nt,ny,nx) end-of-month enthalpy     [J m-2]
    float* runoff_out,     // (nt,ny,nx) pack runoff this month    [kg m-2]
    float* icemelt_out,    // (nt,ny,nx) ice melt this month       [kg m-2]
    float* tsurf_out,      // (nt,ny,nx) surface temperature       [C]
    float* albedo_out,     // (nt,ny,nx) broadband albedo          [-]
    const float* __restrict__ precip,   // (nt,ny,nx) total precipitation       [m a-1 w.e.]
    const float* __restrict__ t2m,      // (nt,ny,nx) 2 m air temperature       [C]
    const float* __restrict__ insol,    // (nt,ny,nx) monthly-mean insolation   [fraction of direct sun]
    const float* __restrict__ t_base,   // (ny,nx)    basal/substrate temperature [C]
    const float* __restrict__ debris,   // (ny,nx)    melt attenuation for snow-free ice [0,1]
    const float* __restrict__ temp_dev, // (nt,n_sub) sub-step air-temperature deviations [C]
    float L_f, float c_i, float c_w,
    float H_atm, float H_base0,
    float q_sw_bulk, float q_sw_insol, float q_lw0,
    float albedo_snow, float albedo_ice, float M_albedo,
    float T_transition, float inv_M_insulation, float M_eps,
    float rho_w, float dt, int glacier_surface,
    int start_month, int ny, int nx, int nt, int n_sub
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ny || j >= nx) return;

    int pix = i * nx + j;
    int slab = ny * nx;
    float Tb = t_base[pix];
    float D = debris[pix];   // multiplies every glacier-ice melt flux (1 = clean ice)

    float M = 0.0f;   // active reservoir mass    [kg m-2], >= 0
    float E = 0.0f;   // active reservoir enthalpy [J m-2], <= 0 when retained

    // Sub-monthly integration: each month is advanced in n_sub explicit sub-steps of
    // length dt_sub, driven by the monthly-mean air temperature plus a per-sub-step
    // deviation temp_dev[m*n_sub + d] (spatially uniform; RNG lives in Python).  This
    // captures the effect of sub-monthly temperature variance on a *stateful* pack --
    // cold-content, refreezing and albedo feedback -- for which no closed-form Gaussian
    // expectation (the PDD Phi/phi trick in smb.cu) exists.  n_sub == 1 with zero
    // deviations reduces exactly to the single monthly step.
    float dt_sub = dt / (float)n_sub;      // dt/1.0f == dt exactly when n_sub == 1

    for (int s = 0; s < nt; s++) {
        int m = (start_month + s) % nt;   // calendar month, water-year order
        int idx = m * slab + pix;

        float M_prev = M;                 // retained mass at start of month [kg m-2]
        float runoff_m = 0.0f;            // pack runoff summed over sub-steps [kg m-2]
        float icemelt_m = 0.0f;           // ice melt summed over sub-steps    [kg m-2]
        float tsurf_sum = 0.0f;           // for the sub-step mean surface temperature
        float alpha_sum = 0.0f;           // for the sub-step mean albedo

        for (int d = 0; d < n_sub; d++) {
            float T_air = t2m[idx] + temp_dev[m * n_sub + d];

            // --- advective mass & enthalpy input, distributed over the sub-step ---
            float dM = fmaxf(precip[idx], 0.0f) * rho_w * dt_sub;    // [kg m-2]
            float f_rain = enth_rain_fraction(T_air, T_transition);
            float h_precip = enth_precip_enthalpy(T_air, f_rain, L_f, c_i, c_w);
            M += dM;
            E += dM * h_precip;

            // --- albedo from *diagnostic solid mass* (liquid drains for the optics) ---
            // liquid = clip(max(E,0)/L_f, 0, M); M_solid = max(M - liquid, 0).
            float liquid = fminf(fmaxf(fmaxf(E, 0.0f) / L_f, 0.0f), M);
            float M_solid = fmaxf(M - liquid, 0.0f);
            float f_snow = 1.0f - expf(-M_solid / M_albedo);        // 0 bare ice -> 1 deep snow
            float alpha = albedo_ice + f_snow * (albedo_snow - albedo_ice);

            // Absorbed shortwave = (1 - alpha) * (constant bulk + seasonal term).  The
            // seasonal term scales q_sw_insol (the clear-sky flux at full direct sun)
            // by the monthly-mean insolation fraction, giving the terrain-shaded,
            // sun-angle-modulated shortwave.  q_sw_insol == 0 recovers the bulk model.
            float q_sw = (1.0f - alpha) * (q_sw_bulk + q_sw_insol * insol[idx]);

            // --- basal conductance (snow insulation) & linear heat exchange ---
            // H_base = H_base0 / (1 + M_solid/M_insulation); inv_M_insulation == 0
            // recovers a constant H_base0 (M_insulation -> inf).
            float H_base = H_base0 / (1.0f + M_solid * inv_M_insulation);
            float H_total = H_atm + H_base;
            float Q0 = q_sw + q_lw0 + H_atm * T_air + H_base * Tb;

            float T_surface;
            float runoff = 0.0f;
            float ice_melt = 0.0f;

            if (M > M_eps) {
                // Backward-Euler cold branch: T_s = E_new / (M c_i).
                float E_cold = (E + dt_sub * Q0) / (1.0f + dt_sub * H_total / (M * c_i));

                if (E_cold <= 0.0f) {
                    E = E_cold;
                    T_surface = E / (M * c_i);
                } else {
                    // Melting branch: hold T_s = 0 C; positive energy becomes runoff
                    // and, after snow exhaustion, ice melt.
                    T_surface = 0.0f;
                    float E_avail = E + dt_sub * Q0;
                    runoff = fminf(E_avail / L_f, M);
                    float excess = E_avail - runoff * L_f;
                    ice_melt = glacier_surface ? D * fmaxf(excess, 0.0f) / L_f : 0.0f;
                    M -= runoff;
                    E = 0.0f;
                }
            } else {
                // No reservoir exists: treat the exposed surface as quasi-steady and
                // do not store negative enthalpy in a zero-mass state.
                float T_eq = (H_total > 0.0f) ? Q0 / H_total : 0.0f;
                if (T_eq < 0.0f) {
                    T_surface = T_eq;   // exact steady balance, no nonadvective flux
                } else {
                    T_surface = 0.0f;
                    ice_melt = glacier_surface ? D * fmaxf(dt_sub * Q0, 0.0f) / L_f : 0.0f;
                }
                M = 0.0f;
                E = 0.0f;
            }

            // Numerical cleanup: collapse a dust-mass reservoir to a clean bare state.
            // Runs every sub-step -- the carried (M,E) feeds the next sub-step's regime
            // selection and albedo, so it must not be deferred to end of month.
            if (M < M_eps) { M = 0.0f; E = 0.0f; }

            runoff_m += runoff;
            icemelt_m += ice_melt;
            tsurf_sum += T_surface;
            alpha_sum += alpha;
        }

        // Surface mass balance rate [m a-1 w.e.]: net change of the retained
        // reservoir mass less glacier-ice melt, per unit time (note dt, not dt_sub).
        // Summing s*dt over the water year telescopes to (M_final - sum ice_melt)/rho_w
        // = accumulation (surviving snow) minus ablation (ice melt).
        smb_out[idx] = (M - M_prev - icemelt_m) / (rho_w * dt);

        M_out[idx] = M;                          // end-of-month state
        E_out[idx] = E;
        runoff_out[idx] = runoff_m;              // fluxes summed over sub-steps
        icemelt_out[idx] = icemelt_m;
        tsurf_out[idx] = tsurf_sum / (float)n_sub;   // diagnostics averaged over sub-steps
        albedo_out[idx] = alpha_sum / (float)n_sub;
    }
}


// =========================================================================== //
// Reverse-mode adjoint of compute_enthalpy.
//
// The forward pass is a chain of sub-steps carrying the state (M, E).  The adjoint
// is its hand-derived vector-Jacobian product: a reverse-in-time scan carrying the
// state adjoints (bar_M, bar_E), exactly as compute_smb_grad carries the snow-depth
// adjoint.  This kernel is the GPU port of the scalar reference `run_column_adjoint`
// in glare/enthalpy.py and reproduces it to floating-point tolerance.
//
// Because the state cannot be reversed analytically through the three regimes, we
// checkpoint (gradient checkpointing, as in compute_smb_grad): a forward replay
// records the month-entry state, and each month's sub-step-entry states are re-derived
// on the fly during the reverse scan.  Each sub-step is then recomputed forward
// (enth_substep, filling an EnthIm) and transposed (enth_substep_back).
//
// Differentiation targets: the forcing (t2m, precip, insol, t_base), the static
// debris melt-attenuation field, and the eight energy-balance parameters (H_atm,
// H_base0, q_sw_bulk, q_sw_insol, q_lw0, albedo_snow, albedo_ice, M_albedo).  The
// thermodynamic constants (L_f, c_i, c_w, T_transition, M_insulation, rho_w) are
// held fixed.  The loss is seeded through the mass/energy
// outputs (smb, M, E, runoff, ice_melt); the pure diagnostics (t_surface, albedo)
// are not seeded.  Parameter gradients are accumulated per pixel and reduced to a
// scalar on the host, matching compute_smb_grad's grad_mf/grad_rf convention.

#define GLARE_ENTH_NT_MAX 64
#define GLARE_ENTH_NSUB_MAX 64

// Fixed (non-differentiated) parameters, bundled to keep the device signatures sane.
struct EnthPar {
    float L_f, c_i, c_w, H_atm, H_base0, q_sw_bulk, q_sw_insol, q_lw0;
    float albedo_snow, albedo_ice, M_albedo, T_transition, inv_M_insulation;
    float M_eps, rho_w;
};

// Every intermediate and branch selector the backward pass needs, so it never has
// to reconstruct which regime a sub-step took.
struct EnthIm {
    float M1, E1, h, dM, P, T_air, Ts, Tl, fr, insol, Tb, D;
    float alpha, S, fsnow, fexp, Msol, Hb, denomH, Htot, Q0;
    float denom, numer, Ecold, Eavail;
    int regime;          // 0 cold, 1 melt-at-0, 2 bare
    int runoff_capped;   // melt: the min() hit M1 (snow-limited runoff)
    int bare_melt;       // bare: Teq >= 0 (surface at 0 C, ice may melt)
    int Msol_pos;        // M1 - liq > 0
    int liq_by_M;        // liq = M1 (min's second argument won)
    float gate;          // 1 if the retained mass survived cleanup, else 0
};

// One forcing sub-step.  Numerically identical to the sub-step in compute_enthalpy;
// also fills *im when non-NULL.  P is the precip field value [m a-1 w.e.]; the
// m-w.e. -> kg conversion (rho_w) is applied here, matching the forward kernel.
__device__ __forceinline__ void enth_substep(
        const EnthPar pr, float M_in, float E_in, float P, float T_air,
        float Tb, float insol_v, float D, float dt, int glacier,
        float* M_out, float* E_out, float* runoff_out, float* icemelt_out,
        EnthIm* im) {
    float dM = fmaxf(P, 0.0f) * pr.rho_w * dt;
    float fr = 1.0f / (1.0f + expf(-T_air / pr.T_transition));
    float Ts = fminf(T_air, 0.0f), Tl = fmaxf(T_air, 0.0f);
    float h = (1.0f - fr) * pr.c_i * Ts + fr * (pr.L_f + pr.c_w * Tl);
    float M1 = M_in + dM;
    float E1 = E_in + dM * h;

    float liq_un = fmaxf(E1, 0.0f) / pr.L_f;
    int liq_by_M = liq_un > M1;
    float liq = liq_by_M ? M1 : liq_un;
    float Msol_raw = M1 - liq;
    int Msol_pos = Msol_raw > 0.0f;
    float Msol = Msol_pos ? Msol_raw : 0.0f;
    float fexp = expf(-Msol / pr.M_albedo);
    float fsnow = 1.0f - fexp;
    float alpha = pr.albedo_ice + fsnow * (pr.albedo_snow - pr.albedo_ice);
    float S = pr.q_sw_bulk + pr.q_sw_insol * insol_v;
    float q_sw = (1.0f - alpha) * S;
    float denomH = 1.0f + Msol * pr.inv_M_insulation;
    float Hb = pr.H_base0 / denomH;
    float Htot = pr.H_atm + Hb;
    float Q0 = q_sw + pr.q_lw0 + pr.H_atm * T_air + Hb * Tb;

    float runoff = 0.0f, ice_melt = 0.0f;
    float denom = 0.0f, numer = 0.0f, Ecold = 0.0f, Eavail = 0.0f;
    int regime, runoff_capped = 0, bare_melt = 0;
    float M2, E2;
    if (M1 > pr.M_eps) {
        denom = 1.0f + dt * Htot / (M1 * pr.c_i);
        numer = E1 + dt * Q0;
        Ecold = numer / denom;
        if (Ecold <= 0.0f) {
            regime = 0; M2 = M1; E2 = Ecold;
        } else {
            regime = 1;
            Eavail = E1 + dt * Q0;
            runoff_capped = (Eavail / pr.L_f) > M1;
            runoff = runoff_capped ? M1 : Eavail / pr.L_f;
            float excess = Eavail - runoff * pr.L_f;
            ice_melt = glacier ? D * fmaxf(excess, 0.0f) / pr.L_f : 0.0f;
            M2 = M1 - runoff; E2 = 0.0f;
        }
    } else {
        regime = 2;
        float Teq = (Htot > 0.0f) ? Q0 / Htot : 0.0f;
        if (Teq >= 0.0f) {
            bare_melt = 1;
            ice_melt = glacier ? D * fmaxf(dt * Q0, 0.0f) / pr.L_f : 0.0f;
        }
        M2 = 0.0f; E2 = 0.0f;
    }
    float gate = (M2 >= pr.M_eps) ? 1.0f : 0.0f;
    if (gate == 0.0f) { M2 = 0.0f; E2 = 0.0f; }

    *M_out = M2; *E_out = E2; *runoff_out = runoff; *icemelt_out = ice_melt;
    if (im) {
        im->M1 = M1; im->E1 = E1; im->h = h; im->dM = dM; im->P = P;
        im->T_air = T_air; im->Ts = Ts; im->Tl = Tl; im->fr = fr;
        im->insol = insol_v; im->Tb = Tb; im->D = D; im->alpha = alpha; im->S = S;
        im->fsnow = fsnow; im->fexp = fexp; im->Msol = Msol; im->Hb = Hb;
        im->denomH = denomH; im->Htot = Htot; im->Q0 = Q0; im->denom = denom;
        im->numer = numer; im->Ecold = Ecold; im->Eavail = Eavail;
        im->regime = regime;
        im->runoff_capped = runoff_capped; im->bare_melt = bare_melt;
        im->Msol_pos = Msol_pos; im->liq_by_M = liq_by_M; im->gate = gate;
    }
}

// Transpose of enth_substep.  Consumes the adjoints of this sub-step's outputs
// (bar_M2/bar_E2 of the state, g_runoff/g_icemelt of the fluxes) and returns the
// input-state adjoints (bar_M_in, bar_E_in) while accumulating (+=) into the eight
// parameter grads and the per-sub-step forcing grads (gt2m, gpre, gins, gtb, gD).
__device__ __forceinline__ void enth_substep_back(
        const EnthPar pr, const EnthIm* im, float dt, int glacier,
        float bar_M2, float bar_E2, float g_runoff, float g_icemelt,
        float* bar_M_in, float* bar_E_in,
        float* gHa, float* gHb0, float* gqb, float* gqi, float* gqlw,
        float* gas, float* gai, float* gMa,
        float* gt2m, float* gpre, float* gins, float* gtb, float* gD) {
    // Cleanup clamp (transpose): kill state-output adjoints if the mass collapsed.
    float aM2 = bar_M2 * im->gate;
    float aE2 = bar_E2 * im->gate;
    float L_f = pr.L_f, c_i = pr.c_i;
    float D = im->D;
    float bar_M1 = 0.0f, bar_E1 = 0.0f, bar_Q0 = 0.0f, bar_Htot = 0.0f;

    int regime = im->regime;
    if (regime == 0) {                          // cold branch
        float denom = im->denom, numer = im->numer, Ecold = im->Ecold;
        float M1 = im->M1, Htot = im->Htot;
        float bar_numer = aE2 / denom;
        float bar_denom = -aE2 * Ecold / denom;
        bar_E1 += bar_numer;
        bar_Q0 += dt * bar_numer;
        bar_Htot += bar_denom * (dt / (M1 * c_i));
        bar_M1 += bar_denom * (-dt * Htot / (M1 * M1 * c_i));
        bar_M1 += aM2;                          // M2 = M1
    } else if (regime == 1) {                   // melt-at-0 branch
        float bar_Eavail;
        if (im->runoff_capped) {                // runoff = M1, ice_melt = D excess/L_f
            bar_M1 += g_runoff;
            float bar_excess = g_icemelt * (glacier ? D / L_f : 0.0f);
            if (glacier)                        // excess > 0 in the capped path
                *gD += g_icemelt * (im->Eavail - im->M1 * L_f) / L_f;
            bar_M1 += bar_excess * (-L_f);
            bar_Eavail = bar_excess;            // M2 = 0 -> aM2 gated out
        } else {                                // runoff = Eavail/L_f, ice_melt = 0
            bar_Eavail = g_runoff * (1.0f / L_f) + aM2 * (-1.0f / L_f);
            bar_M1 += aM2;                      // M2 = M1 - runoff (runoff indep of M1)
        }
        bar_E1 += bar_Eavail;                   // Eavail = E1 + dt Q0
        bar_Q0 += dt * bar_Eavail;
    } else {                                    // bare branch
        if (im->bare_melt && glacier && (dt * im->Q0 > 0.0f)) {
            bar_Q0 += g_icemelt * D * (dt / L_f);   // ice_melt = D dt Q0 / L_f
            *gD += g_icemelt * dt * im->Q0 / L_f;
        }
        // M2 = E2 = 0: no state-adjoint flow through the regime.
    }

    // ---- common pre-regime chain (shared by every branch) -------------------- //
    float alpha = im->alpha, S = im->S, insol = im->insol;
    float Hb = im->Hb, denomH = im->denomH, Msol = im->Msol;
    float invMins = pr.inv_M_insulation;

    // Q0 = q_sw + q_lw0 + H_atm T_air + Hb Tb
    float bar_qsw = bar_Q0;
    *gqlw += bar_Q0;
    *gHa += bar_Q0 * im->T_air;
    float bar_T_air = bar_Q0 * pr.H_atm;
    float bar_Hb = bar_Q0 * im->Tb;
    *gtb += bar_Q0 * Hb;
    // Htot = H_atm + Hb
    *gHa += bar_Htot;
    bar_Hb += bar_Htot;
    // Hb = H_base0 / denomH
    *gHb0 += bar_Hb * (1.0f / denomH);
    float bar_Msol = bar_Hb * (-Hb * invMins / denomH);
    // q_sw = (1 - alpha)(q_sw_bulk + q_sw_insol insol)
    *gqb += bar_qsw * (1.0f - alpha);
    *gqi += bar_qsw * (1.0f - alpha) * insol;
    *gins += bar_qsw * (1.0f - alpha) * pr.q_sw_insol;
    float bar_alpha = -bar_qsw * S;
    // alpha = albedo_ice + fsnow (albedo_snow - albedo_ice)
    *gai += bar_alpha * (1.0f - im->fsnow);
    *gas += bar_alpha * im->fsnow;
    float bar_fsnow = bar_alpha * (pr.albedo_snow - pr.albedo_ice);
    // fsnow = 1 - exp(-Msol / M_albedo)
    bar_Msol += bar_fsnow * (im->fexp / pr.M_albedo);
    *gMa += bar_fsnow * (-im->fexp * Msol / (pr.M_albedo * pr.M_albedo));
    // Msol = max(M1 - liq, 0)
    float bar_liq = 0.0f;
    if (im->Msol_pos) { bar_M1 += bar_Msol; bar_liq = -bar_Msol; }
    // liq = min(max(E1,0)/L_f, M1)
    if (im->liq_by_M) bar_M1 += bar_liq;
    else if (im->E1 > 0.0f) bar_E1 += bar_liq * (1.0f / L_f);

    // M1 = M_in + dM ;  E1 = E_in + dM h
    *bar_M_in = bar_M1;
    *bar_E_in = bar_E1;
    float bar_dM = bar_M1 + bar_E1 * im->h;
    float bar_h = bar_E1 * im->dM;
    // dM = max(P,0) rho_w dt   (P is the precip field value; rho_w folds in here)
    *gpre += (im->P > 0.0f) ? bar_dM * pr.rho_w * dt : 0.0f;
    // h = (1 - fr) c_i min(T,0) + fr (L_f + c_w max(T,0)),  fr = sigmoid(T / width)
    float fr = im->fr;
    float dfr = fr * (1.0f - fr) / pr.T_transition;
    float dh_dT = (1.0f - fr) * c_i * ((im->T_air < 0.0f) ? 1.0f : 0.0f)
                + fr * pr.c_w * ((im->T_air > 0.0f) ? 1.0f : 0.0f)
                + dfr * ((pr.L_f + pr.c_w * im->Tl) - c_i * im->Ts);
    bar_T_air += bar_h * dh_dT;
    *gt2m += bar_T_air;                          // T_air = t2m + dev (dev exogenous)
}

extern "C" __global__ void compute_enthalpy_grad(
    // gradient outputs
    // The cube-sized forcing gradients are deliberately NOT __restrict__:
    // EnthalpyBackwardOperators.compute_gradient routes any of them the
    // caller does not consume (`wanted`) to one shared write-only sink, so
    // they may alias each other. Aliased stores are well-defined; nothing
    // reads these outputs inside the kernel.
    float* grad_t2m,       // (nt,ny,nx)
    float* grad_precip,    // (nt,ny,nx)
    float* grad_insol,     // (nt,ny,nx)
    float* __restrict__ grad_t_base,    // (ny,nx)
    float* __restrict__ grad_debris,    // (ny,nx)
    float* __restrict__ grad_H_atm,     // (ny,nx)  per-pixel; host reduces to a scalar
    float* __restrict__ grad_H_base0,   // (ny,nx)
    float* __restrict__ grad_q_sw_bulk, // (ny,nx)
    float* __restrict__ grad_q_sw_insol,// (ny,nx)
    float* __restrict__ grad_q_lw0,     // (ny,nx)
    float* __restrict__ grad_albedo_snow,// (ny,nx)
    float* __restrict__ grad_albedo_ice,// (ny,nx)
    float* __restrict__ grad_M_albedo,  // (ny,nx)
    // adjoint seeds (d loss / d output)
    const float* __restrict__ grad_smb,      // (nt,ny,nx)
    const float* __restrict__ grad_M,        // (nt,ny,nx)
    const float* __restrict__ grad_E,        // (nt,ny,nx)
    const float* __restrict__ grad_runoff,   // (nt,ny,nx)
    const float* __restrict__ grad_ice_melt, // (nt,ny,nx)
    // forcing (same as compute_enthalpy)
    const float* __restrict__ precip,
    const float* __restrict__ t2m,
    const float* __restrict__ insol,
    const float* __restrict__ t_base,
    const float* __restrict__ debris,
    const float* __restrict__ temp_dev,
    float L_f, float c_i, float c_w,
    float H_atm, float H_base0,
    float q_sw_bulk, float q_sw_insol, float q_lw0,
    float albedo_snow, float albedo_ice, float M_albedo,
    float T_transition, float inv_M_insulation, float M_eps,
    float rho_w, float dt, int glacier_surface,
    int start_month, int ny, int nx, int nt, int n_sub
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ny || j >= nx) return;

    int pix = i * nx + j;
    int slab = ny * nx;
    float Tb = t_base[pix];
    float D = debris[pix];
    float dt_sub = dt / (float)n_sub;
    float inv = 1.0f / (rho_w * dt);

    EnthPar pr;
    pr.L_f = L_f; pr.c_i = c_i; pr.c_w = c_w; pr.H_atm = H_atm;
    pr.H_base0 = H_base0; pr.q_sw_bulk = q_sw_bulk; pr.q_sw_insol = q_sw_insol;
    pr.q_lw0 = q_lw0;
    pr.albedo_snow = albedo_snow; pr.albedo_ice = albedo_ice;
    pr.M_albedo = M_albedo; pr.T_transition = T_transition;
    pr.inv_M_insulation = inv_M_insulation; pr.M_eps = M_eps; pr.rho_w = rho_w;

    // 1) Forward replay -> month-entry state checkpoints.
    float M_start[GLARE_ENTH_NT_MAX];
    float E_start[GLARE_ENTH_NT_MAX];
    {
        float M = 0.0f, E = 0.0f, ro, ic;
        for (int s = 0; s < nt; s++) {
            int m = (start_month + s) % nt;
            int idx = m * slab + pix;
            M_start[s] = M; E_start[s] = E;
            for (int d = 0; d < n_sub; d++) {
                float T_air = t2m[idx] + temp_dev[m * n_sub + d];
                enth_substep(pr, M, E, precip[idx], T_air, Tb, insol[idx], D,
                             dt_sub, glacier_surface, &M, &E, &ro, &ic, 0);
            }
        }
    }

    // Per-pixel parameter grad accumulators (reduced to scalars on the host).
    float gHa = 0.0f, gHb0 = 0.0f, gqb = 0.0f, gqi = 0.0f, gqlw = 0.0f;
    float gas = 0.0f, gai = 0.0f, gMa = 0.0f, gtb = 0.0f, gD = 0.0f;

    // 2) Reverse-in-time scan carrying the state adjoints (bar_M, bar_E).
    float bar_M = 0.0f, bar_E = 0.0f;
    for (int s = nt - 1; s >= 0; s--) {
        int m = (start_month + s) % nt;
        int idx = m * slab + pix;

        // Re-derive this month's sub-step-entry states from the checkpoint.
        float Msub[GLARE_ENTH_NSUB_MAX];
        float Esub[GLARE_ENTH_NSUB_MAX];
        {
            float M = M_start[s], E = E_start[s], ro, ic;
            for (int d = 0; d < n_sub; d++) {
                Msub[d] = M; Esub[d] = E;
                float T_air = t2m[idx] + temp_dev[m * n_sub + d];
                enth_substep(pr, M, E, precip[idx], T_air, Tb, insol[idx], D,
                             dt_sub, glacier_surface, &M, &E, &ro, &ic, 0);
            }
        }

        // Seed the end-of-month state and the two flux outputs for month m.
        bar_M += grad_M[idx] + grad_smb[idx] * inv;   // d smb / d M_end = +inv
        bar_E += grad_E[idx];
        float g_ro = grad_runoff[idx];                 // applied to every sub-step
        float g_ic = grad_ice_melt[idx] - grad_smb[idx] * inv;  // d smb/d ice = -inv

        float gt2m_m = 0.0f, gpre_m = 0.0f, gins_m = 0.0f;
        for (int d = n_sub - 1; d >= 0; d--) {
            EnthIm im;
            float M2, E2, ro, ic;
            float T_air = t2m[idx] + temp_dev[m * n_sub + d];
            enth_substep(pr, Msub[d], Esub[d], precip[idx], T_air, Tb, insol[idx],
                         D, dt_sub, glacier_surface, &M2, &E2, &ro, &ic, &im);

            float nbM, nbE;
            enth_substep_back(pr, &im, dt_sub, glacier_surface,
                              bar_M, bar_E, g_ro, g_ic, &nbM, &nbE,
                              &gHa, &gHb0, &gqb, &gqi, &gqlw, &gas, &gai, &gMa,
                              &gt2m_m, &gpre_m, &gins_m, &gtb, &gD);
            bar_M = nbM; bar_E = nbE;
        }
        grad_t2m[idx] = gt2m_m;      // month m's forcing is used only in month m
        grad_precip[idx] = gpre_m;
        grad_insol[idx] = gins_m;

        // M_prev = start-of-month state; d smb/d M_prev = -inv (couples to month s-1).
        bar_M += -grad_smb[idx] * inv;
    }

    grad_H_atm[pix] = gHa;
    grad_H_base0[pix] = gHb0;
    grad_q_sw_bulk[pix] = gqb;
    grad_q_sw_insol[pix] = gqi;
    grad_q_lw0[pix] = gqlw;
    grad_albedo_snow[pix] = gas;
    grad_albedo_ice[pix] = gai;
    grad_M_albedo[pix] = gMa;
    grad_t_base[pix] = gtb;
    grad_debris[pix] = gD;
}
