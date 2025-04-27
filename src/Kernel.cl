__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void gray_scale(read_only image2d_t srcImg, write_only image2d_t dstImg)
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));

    float4 pixel = read_imagef(srcImg, sampler, coord);

    const float3 mul_value = {0.2989, 0.5870, 0.1140};

    const float gray = dot(pixel.xyz, mul_value);
    pixel.xyz = gray;

    write_imagef(dstImg, coord, pixel);
}