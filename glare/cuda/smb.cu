#define GLARE_NT_MAX 64

__device__ __forceinline__ float phi(float z) {
    return 0.3989f*expf(-0.5f * z * z);
}

__device__ __forceinline__ float Phi(float z) {
    return 0.5f * (1.0f + erff(z * 0.7071f));
}

__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

extern "C" __global__ void compute_smb(
    float* __restrict__ smb,
    float* __restrict__ snow_depth,
    const float* __restrict__ c0,
    const float* __restrict__ cc,
    const float* __restrict__ cs,
    const float* __restrict__ T_mean,
    const float* __restrict__ precip,
    const float* __restrict__ debris,
    float mf, float rf, float delta_T, float sigma_T, float phi0,
    float alpha_snow, float alpha_ice, float snow_scale, float dt,
    int start_month, int ny, int nx, int nt
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ny || j >= nx) return;

    int pix = i * nx + j;
    int slab = ny * nx;
    float omega = 2.0f * 3.1416f / 24.0f;
    float inv_sigma = 1.0f / sigma_T;

    // Debris cover scales the ice (snow-free) melt only; pure snow is
    // unaffected.  Static per-pixel field (an upstream model may supply it).
    float deb = debris[pix];

    // Sequential water-year scan: October (start_month) -> September, snow
    // depth reset to zero each October.  The albedo for month m is selected
    // from the start-of-month snow depth via a smooth sigmoid.
    float d_in = 0.0f;
    for (int s = 0; s < nt; s++) {
        int m = (start_month + s) % nt;
        int idx = m * slab + pix;

        if (s == 0) d_in = 0.0f;   // October reset

        float mu = T_mean[idx];
        float pr = precip[idx];
        float r0 = c0[idx];
        float rc = cc[idx];
        float rs = cs[idx];

        // Snow depth is clamped >= 0 (see compute_smb_grad), so the albedo
        // driver is a one-sided ramp: tanh(d/scale) = 0 for bare ice (d=0),
        // -> 1 for deep snow.
        float sig = tanhf(d_in / snow_scale);
        float alpha = alpha_ice + sig * (alpha_snow - alpha_ice);
        float rf_eff = rf * (1.0f - alpha);

        float ipot_weighted = 0.0f;
        for (int h = 0; h < 24; h++) {
            float phase = omega * h;
            float I_h = r0 + rc * cosf(phase) + rs * sinf(phase);
            float T_h = (mu + delta_T * cosf(phase - phi0)) * inv_sigma;
            ipot_weighted += I_h * Phi(T_h);
        }
        ipot_weighted /= 24.0f;

        float z = mu * inv_sigma;
        float Phiz = Phi(z);
        float phiz = phi(z);
        float pdd = mu * Phiz + sigma_T * phiz;

        float melt = (mf * pdd + rf_eff * ipot_weighted);

        // Debris attenuates the snow-free fraction of melt: factor = 1 for
        // pure snow (sig=1), = deb for exposed ice (sig=0).
        float fac = sig + (1.0f - sig) * deb;
        float melt_eff = melt * fac;

        float snowfall = (1.0f - Phiz) * pr;

        // smb is the true mass balance rate (may be negative -> ice loss);
        // snow depth is a separate state clamped at 0 (no negative snowpack).
        smb[idx] = (snowfall - melt_eff);

        d_in = d_in + (snowfall - melt_eff) * dt;   // end-of-month snow depth
        if (d_in < 0.0f) d_in = 0.0f;               // clamp: no negative snow
        snow_depth[idx] = d_in;
    }
}


extern "C" __global__ void compute_smb_grad(
    float* __restrict__ grad_T_mean,
    float* __restrict__ grad_precip,
    float* __restrict__ grad_mf,
    float* __restrict__ grad_rf,
    float* __restrict__ grad_debris,
    const float* __restrict__ grad_smb,
    const float* __restrict__ c0,
    const float* __restrict__ cc,
    const float* __restrict__ cs,
    const float* __restrict__ T_mean,
    const float* __restrict__ precip,
    const float* __restrict__ debris,
    float mf, float rf, float delta_T, float sigma_T, float phi0,
    float alpha_snow, float alpha_ice, float snow_scale, float dt,
    int start_month, int ny, int nx, int nt
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ny || j >= nx) return;

    int pix = i * nx + j;
    int slab = ny * nx;
    float omega = 2.0f * 3.1416f / 24.0f;
    float inv_sigma = 1.0f / sigma_T;

    float deb = debris[pix];

    float grad_mf_acc = 0.0f;
    float grad_rf_acc = 0.0f;
    float grad_debris_acc = 0.0f;

    // Gradient checkpointing: replay the forward snow-depth recurrence in
    // thread-local memory instead of reading a saved (nt,ny,nx) field.
    // d_start[s] = start-of-month snow depth for water-year step s
    // (0 at the October reset).
    float d_start[GLARE_NT_MAX];
    {
        float d = 0.0f;
        for (int s = 0; s < nt; s++) {
            int m = (start_month + s) % nt;
            int idx = m * slab + pix;

            float d_in = (s == 0) ? 0.0f : d;   // October reset
            d_start[s] = d_in;

            float mu = T_mean[idx];
            float pr = precip[idx];
            float r0 = c0[idx];
            float rc = cc[idx];
            float rs = cs[idx];

            float sig = tanhf(d_in / snow_scale);
            float alpha = alpha_ice + sig * (alpha_snow - alpha_ice);
            float rf_eff = rf * (1.0f - alpha);

            float ipot_weighted = 0.0f;
            for (int h = 0; h < 24; h++) {
                float phase = omega * h;
                float I_h = r0 + rc * cosf(phase) + rs * sinf(phase);
                float T_h = (mu + delta_T * cosf(phase - phi0)) * inv_sigma;
                ipot_weighted += I_h * Phi(T_h);
            }
            ipot_weighted /= 24.0f;

            float z = mu * inv_sigma;
            float Phiz = Phi(z);
            float phiz = phi(z);
            float pdd = mu * Phiz + sigma_T * phiz;

            float melt = mf * pdd + rf_eff * ipot_weighted;
            float fac = sig + (1.0f - sig) * deb;
            float snowfall = (1.0f - Phiz) * pr;

            d = d_in + (snowfall - melt * fac) * dt;   // end-of-month depth
            if (d < 0.0f) d = 0.0f;                    // clamp: no negative snow
        }
    }

    // Reverse-in-time scan over the water year, carrying bar_D = adjoint of
    // the end-of-month snow depth.  Start-of-month snow depth comes from the
    // checkpointed replay above (d_start already encodes the October reset).
    float bar_D = 0.0f;
    for (int s = nt - 1; s >= 0; s--) {
        int m = (start_month + s) % nt;
        int idx = m * slab + pix;

        float d_in = d_start[s];

        float mu = T_mean[idx];
        float pr = precip[idx];
        float r0 = c0[idx];
        float rc = cc[idx];
        float rs = cs[idx];

        float sig = tanhf(d_in / snow_scale);
        float alpha = alpha_ice + sig * (alpha_snow - alpha_ice);
        float rf_eff = rf * (1.0f - alpha);

        float dipot_dT = 0.0f;
        float ipot_weighted = 0.0f;
        for (int h = 0; h < 24; h++) {
            float phase = omega * h;
            float I_h = r0 + rc * cosf(phase) + rs * sinf(phase);
            float T_h = (mu + delta_T * cosf(phase - phi0)) * inv_sigma;
            ipot_weighted += I_h * Phi(T_h);
            dipot_dT += I_h * phi(T_h) * inv_sigma;
        }
        ipot_weighted /= 24.0f;
        dipot_dT /= 24.0f;

        float z = mu * inv_sigma;
        float Phiz = Phi(z);
        float phiz = phi(z);
        float pdd = mu * Phiz + sigma_T * phiz;

        float melt = mf * pdd + rf_eff * ipot_weighted;
        // Debris factor for the snow-free melt fraction (see compute_smb).
        float fac = sig + (1.0f - sig) * deb;
        float melt_eff = melt * fac;

        // Snow-depth clamp: the carried adjoint only flows back through the
        // recurrence when the (unclamped) end-of-month depth was positive.
        // d_raw = pre-clamp depth; relu'(d_raw) = [d_raw > 0].
        float snowfall = (1.0f - Phiz) * pr;
        float d_raw = d_in + (snowfall - melt_eff) * dt;
        float bar_D_eff = (d_raw > 0.0f) ? bar_D : 0.0f;

        // Combined sensitivity of the shared (snowfall - melt_eff) term: it
        // feeds both smb[m] (weight grad_smb) and the snow-depth update
        // (weight dt*bar_D_eff from downstream months, gated by the clamp).
        float g = grad_smb[idx] + dt * bar_D_eff;

        float dacc_dT = -pr * phiz * inv_sigma;
        float dabl_pdd_dT = mf * (Phiz + mu * phiz * inv_sigma - z * phiz);
        float dabl_ins_dT = rf_eff * dipot_dT;

        // melt_eff = fac * melt: the melt sensitivities scale by fac (debris
        // only attenuates the snow-free part; snowfall is unaffected).
        grad_T_mean[idx] = (dacc_dT - fac * (dabl_pdd_dT + dabl_ins_dT)) * g;
        grad_precip[idx] = (1.0f - Phiz) * g;
        grad_mf_acc -= fac * pdd * g;
        grad_rf_acc -= fac * (1.0f - alpha) * ipot_weighted * g;
        // d melt_eff / d debris = melt * d fac / d debris = melt*(1 - sig).
        grad_debris_acc -= melt * (1.0f - sig) * g;

        // Adjoint of the start-of-month snow depth d_in.  d_in enters melt_eff
        // = fac(d_in)*melt(d_in) two ways via the tanh ramp: through the albedo
        // (rf_eff -> melt) and through fac itself.  The +d_in carried through
        // the depth recurrence is gated by the clamp (bar_D_eff).
        float dsig_dd = (1.0f - sig * sig) / snow_scale;   // d/dd tanh(d/scale)
        float dmelt_dd = -rf * ipot_weighted * (alpha_snow - alpha_ice) * dsig_dd;
        float dmelt_eff_dd = fac * dmelt_dd + melt * (1.0f - deb) * dsig_dd;
        float bar_d_in = bar_D_eff + g * (-dmelt_eff_dd);

        bar_D = bar_d_in;   // becomes the carried adjoint for step s-1
                            // (discarded at s == 0: October reset, d_in const)
    }
    grad_mf[pix] = grad_mf_acc;
    grad_rf[pix] = grad_rf_acc;
    grad_debris[pix] = grad_debris_acc;
}
