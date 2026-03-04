extern "C" __global__ 
void azimuth_trace(
    float* __restrict__ max_zenith,  // (n_rays, max_ray_len, 2) - (i,j) pairs
    int* __restrict__ max_j, // (n_rays,)
    int* __restrict__ max_i, // (n_rays,)
    const float* __restrict__ dem,      // (ny, nx)
    float j_basis,
    float i_basis,
    float step_size,                 // horizontal distance per step (meters)
    int nx, int ny
    ) {
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= ny || j >= nx) return;

    float base_elev = dem[i * nx + j];
    
    float pixel_elev = 0.0;
    float pixel_dist = 0.0;

    float current_j = j;
    float current_i = i;

    float maximum_zenith = 0.0f;
    int maximum_j = j;
    int maximum_i = i;

    float zenith = 0.0f;
    int round_j = j;
    int round_i = i;
    while (true) {
	current_j += j_basis*step_size;
        current_i += i_basis*step_size;
        pixel_dist += step_size;

	round_j = roundf(current_j);
	round_i = roundf(current_i);

	if (round_i < 0 || round_i >= ny || round_j < 0 || round_j >= nx) break;

	pixel_elev = dem[round_i*nx + round_j];
        zenith = (pixel_elev - base_elev) / pixel_dist;
	if (zenith > maximum_zenith) { 
	    maximum_zenith = zenith;
	    maximum_j = round_j;
	    maximum_i = round_i;
	}
    }

    max_zenith[i * nx + j] = maximum_zenith;
    max_j[i * nx + j] = maximum_j;
    max_i[i * nx + j] = maximum_i;
}






            

    
     




