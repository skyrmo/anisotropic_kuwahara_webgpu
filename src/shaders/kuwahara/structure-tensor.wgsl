

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba16float, write>;

// Sobel kernels for gradient computation
// Horizontal gradient (Gx)
const sobelX = array<array<f32, 3>, 3>(
    array<f32, 3>(-1.0, 0.0, 1.0),
    array<f32, 3>(-2.0, 0.0, 2.0),
    array<f32, 3>(-1.0, 0.0, 1.0)
);

// Vertical gradient (Gy)
const sobelY = array<array<f32, 3>, 3>(
    array<f32, 3>(-1.0, -2.0, -1.0),
    array<f32, 3>( 0.0,  0.0,  0.0),
    array<f32, 3>( 1.0,  2.0,  1.0)
);

// Convert RGB to luminance (grayscale)
fn rgb2gray(color: vec3f) -> f32 {
    // Using standard luminance weights
    return dot(color, vec3f(0.299, 0.587, 0.114));
}

// Sample texture with clamping to edges
fn sampleLuminance(coords: vec2i, texSize: vec2i) -> f32 {
    let clampedCoords = clamp(coords, vec2i(0), texSize - vec2i(1));
    let color = textureLoad(inputTexture, clampedCoords, 0).rgb;
    return rgb2gray(color);
}

@compute @workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) globalId: vec3u) {
    let texSize = textureDimensions(inputTexture);
    let coords = vec2i(globalId.xy);

    // Boundary check
    if (coords.x >= i32(texSize.x) || coords.y >= i32(texSize.y)) {
        return;
    }

    // Compute Sobel gradients in x and y directions
    var Sx: f32 = 0.0;
    var Sy: f32 = 0.0;

    // Apply Sobel kernels in 3x3 neighborhood
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let sampleCoords = coords + vec2i(dx, dy);
            let luminance = sampleLuminance(sampleCoords, vec2i(texSize));

            // Accumulate gradients
            let kernelX = sobelX[dy + 1][dx + 1];
            let kernelY = sobelY[dy + 1][dx + 1];

            Sx += luminance * kernelX;
            Sy += luminance * kernelY;
        }
    }

    // // Normalize gradients (Sobel kernel sum = 8)
    // // Ignored for now as not in unity
    // Sx /= 8.0;
    // Sy /= 8.0;

    // Compute structure tensor components
    // G = [G_xx  G_xy]
    //     [G_xy  G_yy]
    let G_xx = Sx * Sx;
    let G_yy = Sy * Sy;
    let G_xy = Sx * Sy;

    // Store in output texture
    // Format: (G_xx, G_yy, G_xy, 0.0)
    textureStore(outputTexture, coords, vec4f(G_xx, G_yy, G_xy, 1.0));
}
