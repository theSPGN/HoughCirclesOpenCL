__constant float4 rgb_ntsc_proportion = {0.2989f, 0.5870f, 0.1140f, 0.0f};

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void hough_transform(read_only image2d_t input_image,
                              write_only image2d_t output_image)
{
  // ***
  // Read input picture and changing it to gray scale
  // ***

  // Read (x, y) coordinates of buffer
  int2 coord = (int2)(get_global_id(0), get_global_id(1));

  // Read size of image (width, height)
  uint2 image_size = (uint2)(get_global_size(0), get_global_size(1));

  // Read pixel and converting itself to float4 vector
  float4 pixel = convert_float4(read_imageui(input_image, sampler, coord));

  // Changing picture to gray scale
  float4 pixel_gray_out = dot(pixel, rgb_ntsc_proportion);


  // ***
  // Performing Sobel operation
  // ***

  // Sobel kernel for x derivative
  const int sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

  // Sobel kernel for y derivative
  const int sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

  // Initialize gradients
  float2 gradient = (float2)(0, 0);

  // Apply Sobel filter using vector operations
  const int2 offsets[9] = {(int2)(-1, -1), (int2)(0, -1), (int2)(1, -1),
                           (int2)(-1, 0),  (int2)(0, 0),  (int2)(1, 0),
                           (int2)(-1, 1),  (int2)(0, 1),  (int2)(1, 1)};

  // Initialize variables before loop
  int2 neighbor_coord = (int2)(0);
  float4 neighbor_pixel = (float4)(0);
  float neighbor_gray = 0.0f;

  for (uint i = 0; i < 9; i++) {

    // Read  (x, y) coordinates of neighbor pixel
    neighbor_coord = coord + offsets[i];

    // Check bounds (if exceeded then go to next iteration)
    if (neighbor_coord.x < 0 || neighbor_coord.x > image_size.x ||
        neighbor_coord.y < 0 || neighbor_coord.y > image_size.y)
      continue;

    // Read neighbor pixel and convert to gray scale
    neighbor_pixel = convert_float4(read_imageui(input_image, sampler, neighbor_coord));

    // In separate kernels shall be deleted:
    neighbor_gray = dot(neighbor_pixel, rgb_ntsc_proportion);

    // Accumulate gradients
    gradient += neighbor_gray * (float2)(sobel_x[i], sobel_y[i]);
  }

  // Compute gradient magnitude
  float gradient_magnitude = sqrt(dot(gradient, gradient));

  // Assign gradient magnitude to pixel
  pixel = (float4)((float3)(gradient_magnitude), 1.0f);

  // Binarization of image
  // float binarization_edge = 120.0f;
  // pixel = step(binarization_edge, pixel) * 255.0f;

  // Write pixel to output buffer
  write_imageui(output_image, coord, convert_uint4(pixel * 255));
}