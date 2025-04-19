__constant float4 rgb_ntsc_proportion = {0.2989f, 0.5870f, 0.1140f, 0.0f};

__kernel void hough_transform(__global uchar4 *input_image,
                              __global uchar4 *output_image) {

  // ***
  // Read input picture and changing it to gray scale
  // ***

  // Read (x, y) coordinates of buffer
  int2 coord = (int2)(get_global_id(0), get_global_id(1));

  // Read size of image (width, height)
  uint2 image_size = (uint2)(get_global_size(0), get_global_size(1));

  // Change relative coordinates of buffer to global index of
  // Pixel in input image
  uint pixel_idx = coord.x + coord.y * image_size.x;

  // Read pixel and converting itself to float4 vector
  float4 pixel = convert_float4(input_image[pixel_idx]);

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

  for (uint i = 0; i < 9; i++) {

    // Read  (x, y) coordinates of neighbor pixel
    int2 neighbor_coord = coord + offsets[i];

    // Check bounds (if exceeded then go to next iteration)
    if (neighbor_coord.x < 0 || neighbor_coord.x > image_size.x ||
        neighbor_coord.y < 0 || neighbor_coord.y > image_size.y)
      continue;

    // Read neighbor pixel global coords
    uint neighbor_idx = neighbor_coord.x + neighbor_coord.y * image_size.x;

    // Read neighbor pixel and convert to gray scale
    float4 neighbor_pixel = convert_float4(input_image[neighbor_idx]);
    // In separate kernels shall be deleted:
    float neighbor_gray = dot(neighbor_pixel, rgb_ntsc_proportion);

    // Accumulate gradients
    gradient += neighbor_gray * (float2)(sobel_x[i], sobel_y[i]);
  }

  // Compute gradient magnitude
  float gradient_magnitude = sqrt(dot(gradient, gradient));

  // Assign gradient magnitude to pixel
  pixel = (float4)((float3)(gradient_magnitude), 1.0f);

  // Write pixel to output buffer
  output_image[pixel_idx] = convert_uchar4(pixel);
}