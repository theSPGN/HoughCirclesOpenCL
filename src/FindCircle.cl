__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;


#define LOCAL_GROUP_SIZE 64

__kernel void FindCircle(read_only  image2d_t input_image,
                        __global const int *circle_x_pos,
                        __global const int *circle_y_pos,
                                 const int length,
                         write_only image2d_t output_image)
{
    const int3 coord = (int3)(get_global_id(0), get_global_id(1), get_local_id(2));
    const int2 image_size = (int2)(get_global_size(0), get_global_size(1));

    const int part_length = length / LOCAL_GROUP_SIZE;
    const int start_idx = part_length * coord.z;

    const uint4 center_pixel = read_imageui(input_image, sampler, coord.xy);

    __local uint local_accumulator[LOCAL_GROUP_SIZE];

    uint accumulator = 0;

    for (int i = start_idx; i < start_idx + part_length; ++i)
    {
        const int2 curr_coord = coord.xy + (int2)(circle_x_pos[i], circle_y_pos[i]);

        if (curr_coord.x >= 0 && curr_coord.x < image_size.x && curr_coord.y >= 0 && curr_coord.y < image_size.y)
        {
            const uint4 curr_pixel = read_imageui(input_image, sampler, curr_coord);
            accumulator += (uint)(curr_pixel.x != 0);
        }
    }

    local_accumulator[get_local_id(2)] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (coord.z != 0)
        return;

    #pragma unroll
    for (int i = 1; i < LOCAL_GROUP_SIZE; ++i)
    {
        accumulator += local_accumulator[i];
    }
    write_imageui(output_image, coord.xy,  (uint4)(accumulator, 0, 0, 255));
}


__kernel void FindRadius(read_only image2d_t input_image,
                         uint threshold,
                         int hough_max_radius,
                         write_only image2d_t output_image)
{
    const int2 coord = (int2)(get_global_id(0), get_global_id(1));
    const uint4 center_pixel = read_imageui(input_image, sampler, coord);
    uint4 output = (uint4)(0);

    bool has_larger_neighbor = false;

    if (center_pixel.x > threshold)
    {
        for (int dy = -hough_max_radius; dy <= hough_max_radius; ++dy)
        {
            for (int dx = -hough_max_radius; dx <= hough_max_radius; ++dx) {
                int2 neighbor_coord = coord + (int2)(dx, dy);
                uint4 neighbor_pixel = read_imageui(input_image, sampler, neighbor_coord);

                if (neighbor_pixel.x > center_pixel.x) {
                    has_larger_neighbor = true;
                    break;
                }
            }
            if (has_larger_neighbor) break; 
        }
        if (!has_larger_neighbor) {
            output = (uint4)(255);
        }
    }
    write_imageui(output_image, coord, output);
}
