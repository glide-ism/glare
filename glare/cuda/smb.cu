__device__ __forceinline__ float phi(float z) {
    return 0.3989f*expf(-0.5f * z * z);
}

__device__ __forceinline__ float Phi(float z) {
    return 0.5f * (1.0f + erff(z * 0.7071f));
}

extern "C" __global__ void compute_smb(
    float* __restrict__ smb,
    const float* __restrict__ c0,
    const float* __restrict__ cc,
    const float* __restrict__ cs,
    const float* __restrict__ T_mean,
    const float* __restrict__ precip,
    float mf, float rf, float delta_T, float sigma_T, float phi0,
    int ny, int nx, int nt
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ny || j >= nx) return;

    int pix = i * nx + j;
    int slab = ny * nx;
    float omega = 2.0f * 3.1416f / 24.0f;
    float inv_sigma = 1.0f / sigma_T;

    for (int m = 0; m < nt; m++) {
        int idx = m * slab + pix;
        float mu = T_mean[idx];
	float pr = precip[idx];
        float r0 = c0[idx];
        float rc = cc[idx];
        float rs = cs[idx];

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

        float melt = (mf * pdd + rf * ipot_weighted);

        float snowfall = (1.0f - Phiz) * pr;

        smb[idx] = (snowfall - melt);
    }
}


extern "C" __global__ void compute_smb_grad(
    float* __restrict__ grad_T_mean,
    float* __restrict__ grad_precip,
    float* __restrict__ grad_mf,
    float* __restrict__ grad_rf,
    const float* __restrict__ grad_smb,
    const float* __restrict__ c0,
    const float* __restrict__ cc,
    const float* __restrict__ cs,
    const float* __restrict__ T_mean,
    const float* __restrict__ precip,
    float mf, float rf, float delta_T, float sigma_T, float phi0,
    int ny, int nx, int nt
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ny || j >= nx) return;

    int pix = i * nx + j;
    int slab = ny * nx;
    float omega = 2.0f * 3.1416f / 24.0f;
    float inv_sigma = 1.0f / sigma_T;


    float grad_mf_acc = 0.0f;
    float grad_rf_acc = 0.0f;
    for (int m = 0; m < nt; m++) {
        int idx = m * slab + pix;
        float mu = T_mean[idx];
	float pr = precip[idx];
        float r0 = c0[idx];
        float rc = cc[idx];
        float rs = cs[idx];
        float g_smb = grad_smb[idx];

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

        float dacc_dT = -pr * phiz * inv_sigma;
	float dabl_pdd_dT = mf * (Phiz + mu*phiz*inv_sigma - z*phiz);
        float dabl_ins_dT = rf * dipot_dT;

        grad_T_mean[idx] = (dacc_dT - dabl_pdd_dT - dabl_ins_dT) * g_smb;
	grad_precip[idx] = (1.0f - Phiz) * g_smb;
	grad_mf_acc -= pdd * g_smb;
	grad_rf_acc -= ipot_weighted * g_smb;

    }
    grad_mf[pix] = grad_mf_acc;
    grad_rf[pix] = grad_rf_acc;

}

