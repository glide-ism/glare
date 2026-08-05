// Avalanche redistribution operator.
//
// Mass-conserving multiple-flow-direction (MFD) snow routing realised as a fixed
// linear map q = R s, applied to a (already partitioned) solid-precipitation field.
// Because the terrain is static, R is time-invariant and its adjoint is exactly
// R^T -- no tape, no checkpointing.
//
// Routing topology is a pure function of the fixed DEM and the (fixed) deposition
// parameters theta = (s_crit, w_trans, p), so we recompute it in-kernel from a
// read-only 3x3 DEM patch every pass rather than materialising neighbour
// lists/weights (the grid is structured -> adjacency is implicit; these kernels are
// bandwidth bound and the arithmetic is effectively free).
//
// Per cell j the mobile snow is split: a fraction d(slope_j) deposits locally and
// the remainder (1 - d) routes to downslope neighbours with weights
// w_k proportional to (dz/L)^p, normalised over j's downslope set.  d depends on
// slope only (not snow amount), which is what keeps the operator linear in s.

#define AVA_RAD2DEG 57.29577951308232f

__device__ __forceinline__ float ava_sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Recompute cell j's deposition fraction and the 8 downslope outflow weights a_k
// (already scaled by the mobile fraction 1-d, so deposit + sum(a_k) == 1) from its
// 3x3 DEM patch.  off-grid / upslope neighbours get weight 0.  Pits (no downslope
// neighbour) deposit everything locally so mass is conserved.
//
// di/dj index the 8 neighbours in a fixed order; the caller reuses the same order
// to map weights back to neighbour cells, so forward (scatter) and adjoint (gather)
// see identical topology -> the gather is the exact transpose of the scatter.
__device__ __constant__ int AVA_DI[8] = {-1, -1, -1,  0,  0,  1,  1,  1};
__device__ __constant__ int AVA_DJ[8] = {-1,  0,  1, -1,  1, -1,  0,  1};

__device__ void ava_routing(
    const float* __restrict__ srf,
    int i, int j, int ny, int nx,
    float dx, float s_crit, float w_trans, float p,
    float a[8], float* deposit_out
) {
    float z0 = srf[i * nx + j];
    float diag = dx * 1.41421356f;

    float g[8];
    float sum_g = 0.0f;
    float steepest = 0.0f;          // steepest descent tangent (>= 0)

    for (int k = 0; k < 8; k++) {
        int ni = i + AVA_DI[k];
        int nj = j + AVA_DJ[k];
        g[k] = 0.0f;
        if (ni < 0 || ni >= ny || nj < 0 || nj >= nx) continue;   // off-grid

        float L = (AVA_DI[k] != 0 && AVA_DJ[k] != 0) ? diag : dx;
        float dz = z0 - srf[ni * nx + nj];                        // >0 -> downslope
        if (dz <= 0.0f) continue;                                 // not downslope

        float tan_slope = dz / L;
        if (tan_slope > steepest) steepest = tan_slope;
        float gk = powf(tan_slope, p);
        g[k] = gk;
        sum_g += gk;
    }

    float slope_deg = atanf(steepest) * AVA_RAD2DEG;
    float d = ava_sigmoidf((s_crit - slope_deg) / w_trans);

    if (sum_g > 0.0f) {
        float scale = (1.0f - d) / sum_g;
        for (int k = 0; k < 8; k++) a[k] = g[k] * scale;
        *deposit_out = d;
    } else {
        // Local pit / no downslope neighbour: deposit all mobile snow here.
        for (int k = 0; k < 8; k++) a[k] = 0.0f;
        *deposit_out = 1.0f;
    }
}

// One forward (scatter) pass: deposit d*m locally into q (own-cell write, no atomic
// needed) and scatter the routed mobile fraction into the downslope neighbours of
// the next-pass buffer with atomicAdd.
extern "C" __global__ void avalanche_scatter(
    const float* __restrict__ srf,
    const float* __restrict__ mobile_in,
    float* __restrict__ mobile_out,
    float* __restrict__ q,
    float dx, float s_crit, float w_trans, float p,
    int ny, int nx
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ny || j >= nx) return;

    int pix = i * nx + j;
    float m = mobile_in[pix];

    float a[8];
    float deposit;
    ava_routing(srf, i, j, ny, nx, dx, s_crit, w_trans, p, a, &deposit);

    q[pix] += deposit * m;

    for (int k = 0; k < 8; k++) {
        if (a[k] == 0.0f) continue;
        int ni = i + AVA_DI[k];
        int nj = j + AVA_DJ[k];
        atomicAdd(&mobile_out[ni * nx + nj], a[k] * m);
    }
}

// One adjoint (gather) pass -- the exact transpose of avalanche_scatter.
//   bar_out[j] = deposit_j * grad_q[j] + sum_k a_{k,j} * bar_in[neighbour_k(j)]
// Pure 3x3 gather using cell j's own outflow weights, so it is deterministic
// (bitwise reproducible) -- which keeps the FP32 adjoint dot-product test sharp.
extern "C" __global__ void avalanche_gather(
    const float* __restrict__ srf,
    const float* __restrict__ bar_in,
    float* __restrict__ bar_out,
    const float* __restrict__ grad_q,
    float dx, float s_crit, float w_trans, float p,
    int ny, int nx
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ny || j >= nx) return;

    int pix = i * nx + j;

    float a[8];
    float deposit;
    ava_routing(srf, i, j, ny, nx, dx, s_crit, w_trans, p, a, &deposit);

    float acc = deposit * grad_q[pix];
    for (int k = 0; k < 8; k++) {
        if (a[k] == 0.0f) continue;
        int ni = i + AVA_DI[k];
        int nj = j + AVA_DJ[k];
        acc += a[k] * bar_in[ni * nx + nj];
    }
    bar_out[pix] = acc;
}
