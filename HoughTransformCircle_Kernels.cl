__constant sampler_t imageSampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void hough_transform(__read_only image2d_t inputImage,
                              __write_only image2d_t outputImage) {

  int2 coord = (int2)(get_global_id(0), get_global_id(1));

  float4 pixel = convert_float4(
      read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));

  write_imageui(outputImage, coord, convert_uint4(pixel));
}