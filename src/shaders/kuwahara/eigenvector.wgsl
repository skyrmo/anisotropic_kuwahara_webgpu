@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba16float, write>;

// Small epsilon to prevent division by zero
const EPSILON: f32 = 1e-10;

// Compute eigenvalues and eigenvectors of the 2x2 structure tensor
// Input structure tensor format:
//   G_xx (r), G_yy (g), G_xy (b)
//
// Structure tensor matrix:
//   [ G_xx  G_xy ]
//   [ G_xy  G_yy ]
//
// For a 2x2 symmetric matrix, eigenvalues are:
//   λ₁ = (G_xx + G_yy)/2 + sqrt(((G_xx - G_yy)/2)² + G_xy²)
//   λ₂ = (G_xx + G_yy)/2 - sqrt(((G_xx - G_yy)/2)² + G_xy²)
//
// The eigenvector corresponding to λ₁ (major axis) has angle:
//   θ = 0.5 * atan2(2*G_xy, G_xx - G_yy)

struct EigenvectorData {
    lambda1: f32,      // Major eigenvalue
    lambda2: f32,      // Minor eigenvalue
    theta: f32,        // Angle of major eigenvector
    anisotropy: f32,   // Measure of directionality
}

fn computeEigenvalues(G_xx: f32, G_yy: f32, G_xy: f32) -> vec2f {
    // Trace and determinant of structure tensor
    let trace = G_xx + G_yy;
    let det = G_xx * G_yy - G_xy * G_xy;

    // Discriminant for eigenvalue formula
    // trace²/4 - det = ((G_xx + G_yy)/2)² - (G_xx*G_yy - G_xy²)
    //                = ((G_xx - G_yy)/2)² + G_xy²
    let discriminant = trace * trace * 0.25 - det;

    // Handle numerical precision issues
    let sqrtDisc = sqrt(max(discriminant, 0.0));

    let halfTrace = trace * 0.5;
    let lambda1 = halfTrace + sqrtDisc;  // Larger eigenvalue
    let lambda2 = halfTrace - sqrtDisc;  // Smaller eigenvalue

    return vec2f(lambda1, lambda2);
}

fn computeEigenvectorAngle(G_xx: f32, G_yy: f32, G_xy: f32) -> f32 {
    // Angle of major eigenvector
    // θ = 0.5 * atan2(2*G_xy, G_xx - G_yy)
    let numerator = 2.0 * G_xy;
    let denominator = G_xx - G_yy;

    return 0.5 * atan2(numerator, denominator);
}

fn computeAnisotropy(lambda1: f32, lambda2: f32) -> f32 {
    // Anisotropy measure: (λ₁ - λ₂) / (λ₁ + λ₂)
    // Returns 0 for isotropic (circular) regions
    // Returns 1 for highly anisotropic (linear) regions
    let sum = lambda1 + lambda2;
    let diff = lambda1 - lambda2;

    // Prevent division by zero
    if (sum < EPSILON) {
        return 0.0;
    }

    return clamp(diff / (sum + EPSILON), 0.0, 1.0);
}

fn analyzeStructureTensor(G_xx: f32, G_yy: f32, G_xy: f32) -> EigenvectorData {
    var result: EigenvectorData;

    // Compute eigenvalues
    let eigenvalues = computeEigenvalues(G_xx, G_yy, G_xy);
    result.lambda1 = eigenvalues.x;
    result.lambda2 = eigenvalues.y;

    // Compute eigenvector angle
    result.theta = computeEigenvectorAngle(G_xx, G_yy, G_xy);

    // Compute anisotropy
    result.anisotropy = computeAnisotropy(result.lambda1, result.lambda2);

    return result;
}

@compute @workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) globalId: vec3u) {
    let texSize = textureDimensions(inputTexture);
    let coords = vec2i(globalId.xy);

    // Boundary check
    if (coords.x >= i32(texSize.x) || coords.y >= i32(texSize.y)) {
        return;
    }

    // Load smoothed structure tensor components
    let tensor = textureLoad(inputTexture, coords, 0);
    let G_xx = tensor.r;
    let G_yy = tensor.g;
    let G_xy = tensor.b;

    // Analyze structure tensor to get eigenvalues and eigenvectors
    let eigenData = analyzeStructureTensor(G_xx, G_yy, G_xy);

    // Store results in output texture
    // Format: (cos(θ), sin(θ), anisotropy, λ₁)
    // Using cos/sin instead of angle avoids trigonometry in next stage
    let cosTheta = cos(eigenData.theta);
    let sinTheta = sin(eigenData.theta);

    let output = vec4f(
        cosTheta,              // r: X component of major eigenvector
        sinTheta,              // g: Y component of major eigenvector
        eigenData.anisotropy,  // b: Anisotropy measure [0,1]
        eigenData.lambda1      // a: Major eigenvalue (for debugging/visualization)
    );

    textureStore(outputTexture, coords, output);
}
