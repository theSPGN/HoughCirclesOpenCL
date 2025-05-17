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


__kernel void FindCircle2(read_only  image2d_t input_image,
                              const  int base_radius,
                              const  int tolerance,
                           __global  uint* accumulator)
{
    const int2 image_size = (int2)(get_global_size(0), get_global_size(1));
    const int2 coord = (int2)(get_global_id(0), get_global_id(1));

    const uint4 center_pixel = read_imageui(input_image, sampler, coord.xy);


    if (center_pixel.x == 0)
        return;

    for (int angle = 0; angle < 360; angle += 2)
    {
        const float theta = radians((float)angle);
        const float cos_value = cos(theta);
        const float sin_value = sin(theta);

        for (int dr = -tolerance; dr <= tolerance; ++dr)
        {
            const int radius = base_radius + dr;

            int a = (int)(coord.x - radius * cos_value);
            int b = (int)(coord.y - radius * sin_value);

            if (a >= 0 && a < image_size.x && b >= 0 && b < image_size.y)
            {
                int idx = b * image_size.x + a;
                atomic_inc(accumulator + idx);
            }
        }
    }

    /* TODO: add loop over radius
    for (int r_idx = 0; r_idx < num_radii; ++r_idx)
    {
        int r = radii[r_idx];

        for (int angle = 0; angle < 360; angle += 5)
        {
            float theta = radians((float)angle);
            int a = (int)(coord.x - r * cos(theta));
            int b = (int)(coord.y - r * sin(theta));

            if (a >= 0 && a < image_size.x && b >= 0 && b < image_size.y)
            {
                int idx = (r_idx * width * height) + (b * width + a);
                atomic_inc(&accumulator[idx]);
            }
        }
    }
    */
}



__kernel void FindRadius(__global uint* input_image,
                         uint threshold,
                         int hough_max_radius,
                         write_only image2d_t output_image)
{
    const int2 image_size = (int2)(get_global_size(0), get_global_size(1));
    const int2 coord = (int2)(get_global_id(0), get_global_id(1));

    const int idx = coord.y * image_size.x + coord.x;
    const uint center_pixel = input_image[idx];
    uint4 output = (uint4)(0);

    bool has_larger_neighbor = false;

    if (center_pixel > threshold)
    {
        for (int dy = -hough_max_radius; dy <= hough_max_radius; ++dy)
        {
            for (int dx = -hough_max_radius; dx <= hough_max_radius; ++dx) {
                int2 neighbor_coord = coord + (int2)(dx, dy);

                if (neighbor_coord.x < 0 || neighbor_coord.x >= image_size.x ||
                    neighbor_coord.y < 0 || neighbor_coord.y >= image_size.y)
                    continue;

                int neighbor_idx = neighbor_coord.y * image_size.x + neighbor_coord.x;
                uint neighbor_pixel = input_image[neighbor_idx];

                if (neighbor_pixel > center_pixel) {
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
