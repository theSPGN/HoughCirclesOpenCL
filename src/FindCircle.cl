__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void FindCircle(read_only  image2d_t input_image,
                                    int       min_radius,
                                    int       max_radius,
                         write_only image2d_t output_image)
{
    const int2 coord = (int2)(get_global_id(0), get_global_id(1));

    // Center point of circle
    const uint4 center_pixel = read_imageui(input_image, sampler, coord);

    uint accumulator = 0;

    for (int dx = -max_radius; dx <= max_radius; ++dx)
    {
        for (int dy = -max_radius; dy <= max_radius; ++dy)
        {
            int distance = dx*dx + dy*dy;
            if (distance < min_radius*min_radius || distance > max_radius*max_radius)
                continue;

            const int2 curr_coord = (int2)(coord.x + dx, coord.y + dy);

            const uint4 curr_pixel = read_imageui(input_image, sampler, curr_coord);

            accumulator += (uint)(curr_pixel.x != 0);
        }
    }
    write_imageui(output_image, coord, accumulator);
}
