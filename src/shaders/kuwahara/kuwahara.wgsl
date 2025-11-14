// Anisotropic Kuwahara-style compute shader (WGSL, corrected)
// Bindings: 0 = original texture (read), 1 = eigenvector texture (read),
// 2 = output storage texture (write), 3 = uniform params.

@group(0) @binding(0) var originalTexture: texture_2d<f32>;
@group(0) @binding(1) var eigenvectorTexture: texture_2d<f32>;
@group(0) @binding(2) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> kuwaharaParams: KuwaharaParams;

struct KuwaharaParams {
    kernelSize: i32,
    sharpness: f32,
    hardness: f32,
    alpha: f32,
    zeroCrossing: f32,
    zeta: f32,
    sigma: f32,
}

const PI: f32 = 3.14159265358979323846;
const SQRT2_OVER_2: f32 = 0.70710678118654752440; // sqrt(2)/2
const EPS: f32 = 1e-6;

struct SectorStatistics {
    colorSum: vec3<f32>,
    colorSqSum: vec3<f32>,
    weightSum: f32,
};

// Sample texture with boundary clamping
fn sampleTexture(coords: vec2<i32>, texSize: vec2<i32>) -> vec3<f32> {
    let clampedCoords = clamp(coords, vec2<i32>(0), texSize - vec2<i32>(1));
    return textureLoad(originalTexture, clampedCoords, 0).xyz;
}

// Compute polynomial sector weights (Unity-like formulation)
// Input: v in transformed (unit-circle) space, zeta, eta
fn computePolynomialWeights(v: vec2<f32>, zeta: f32, eta: f32) -> array<f32, 8> {
    var weights: array<f32, 8>;
    // initialize explicitly
    for (var i: i32 = 0; i < 8; i = i + 1) {
        weights[i] = 0.0;
    }

    var z: f32 = 0.0;
    var vxx: f32 = zeta - eta * v.x * v.x;
    var vyy: f32 = zeta - eta * v.y * v.y;

    // Cardinal directions (ordered 0=N,1=NE,2=W,3=NW,4=S,5=SW,6=E,7=SE) - preserves original mapping
    // Sector 0: North (+Y)
    z = max(0.0, v.y + vxx);
    weights[0] = z * z;
    // Sector 2: West (-X)
    z = max(0.0, -v.x + vyy);
    weights[2] = z * z;
    // Sector 4: South (-Y)
    z = max(0.0, -v.y + vxx);
    weights[4] = z * z;
    // Sector 6: East (+X)
    z = max(0.0, v.x + vyy);
    weights[6] = z * z;

    // Rotate coordinates by +45° for diagonal sectors
    let rotated_v = SQRT2_OVER_2 * vec2<f32>(v.x - v.y, v.x + v.y);
    vxx = zeta - eta * rotated_v.x * rotated_v.x;
    vyy = zeta - eta * rotated_v.y * rotated_v.y;

    // Sector 1: Northeast
    z = max(0.0, rotated_v.y + vxx);
    weights[1] = z * z;
    // Sector 3: Northwest
    z = max(0.0, -rotated_v.x + vyy);
    weights[3] = z * z;
    // Sector 5: Southwest
    z = max(0.0, -rotated_v.y + vxx);
    weights[5] = z * z;
    // Sector 7: Southeast
    z = max(0.0, rotated_v.x + vyy);
    weights[7] = z * z;

    return weights;
}

@compute @workgroup_size(8, 8, 1)
fn computeMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
    // Get texture size and integer pixel coords
    let texSize_u: vec2<u32> = textureDimensions(originalTexture);
    let texSize_i: vec2<i32> = vec2<i32>(i32(texSize_u.x), i32(texSize_u.y));
    let coords: vec2<i32> = vec2<i32>(i32(globalId.x), i32(globalId.y));

    // Bounds check
    if (coords.x < 0 || coords.y < 0 || coords.x >= texSize_i.x || coords.y >= texSize_i.y) {
        return;
    }

    // Read eigenvector texture: expected layout (cosTheta, sinTheta, anisotropy, lambda1)
    let eigenData4 = textureLoad(eigenvectorTexture, coords, 0);
    let cosTheta: f32 = eigenData4.x;
    let sinTheta: f32 = eigenData4.y;
    let anisotropy: f32 = eigenData4.z;
    // w component (lambda1) is ignored here but available as eigenData4.w

    // Safe alpha to avoid division by zero
    let alpha: f32 = kuwaharaParams.alpha;
    let safeAlpha: f32 = max(alpha, EPS);

    // Use sigma parameter directly as the radius
    let kernelRadius: f32 = f32(kuwaharaParams.kernelSize) * 0.5;

    // Compute ellipse axes (a,b) using the formulas you had (with safety)
    // Note: we clamp denominators to avoid division by zero
    let denomAB = max(alpha + anisotropy, EPS);
    var a: f32 = kernelRadius * clamp((alpha + anisotropy) / safeAlpha, 0.1, 2.0);
    var b: f32 = kernelRadius * clamp(safeAlpha / denomAB, 0.1, 2.0);

    // If a or b is extremely small, clamp to EPS to avoid infinities
    a = max(a, 1e-3);
    b = max(b, 1e-3);

    // Rotation matrix R = [ [cos, -sin], [sin, cos] ]
    let R00: f32 = cosTheta;
    let R01: f32 = -sinTheta;
    let R10: f32 = sinTheta;
    let R11: f32 = cosTheta;

    // Scaling matrix S to map ellipse to unit circle: use 1.0/a and 1.0/b
    let S00: f32 = 1.0 / a;
    let S11: f32 = 1.0 / b;

    // Combined transform SR = S * R  (applied to pixel offset vector)
    let SR00: f32 = S00 * R00;
    let SR01: f32 = S00 * R01;
    let SR10: f32 = S11 * R10;
    let SR11: f32 = S11 * R11;

    // Compute conservative loop bounds in pixel space (ceil) to iterate over bounding box of ellipse
    let cos_phi_sq: f32 = cosTheta * cosTheta;
    let sin_phi_sq: f32 = sinTheta * sinTheta;
    let max_x_f: f32 = ceil(sqrt(a * a * cos_phi_sq + b * b * sin_phi_sq));
    let max_y_f: f32 = ceil(sqrt(a * a * sin_phi_sq + b * b * cos_phi_sq));
    let max_x: i32 = i32(max_x_f);
    let max_y: i32 = i32(max_y_f);

    // Compute eta parameter safely for polynomial weights
    let zeta: f32 = kuwaharaParams.zeta;
    let zeroCrossing: f32 = kuwaharaParams.zeroCrossing;
    let sinZeroCross: f32 = sin(zeroCrossing);
    var eta: f32 = 0.0;
    if (abs(sinZeroCross) > EPS) {
        eta = (zeta + cos(zeroCrossing)) / (sinZeroCross * sinZeroCross);
    } else {
        // fallback: pick a small default to avoid division by zero
        eta = zeta; // reasonable fallback (tunable)
    }

    // Initialize sector statistics
    var sectors: array<SectorStatistics, 8>;
    for (var i: i32 = 0; i < 8; i = i + 1) {
        sectors[i].colorSum = vec3<f32>(0.0, 0.0, 0.0);
        sectors[i].colorSqSum = vec3<f32>(0.0, 0.0, 0.0);
        sectors[i].weightSum = 0.0;
    }

    // Sample elliptical neighborhood (iterate bounding box, test mapped unit circle)
    for (var y: i32 = -max_y; y <= max_y; y = y + 1) {
        for (var x: i32 = -max_x; x <= max_x; x = x + 1) {
            // transform pixel offset (x,y) into v = SR * [x,y]
            let xf: f32 = f32(x);
            let yf: f32 = f32(y);
            let v: vec2<f32> = vec2<f32>(
                SR00 * xf + SR01 * yf,
                SR10 * xf + SR11 * yf
            );

            // Only accept points within unit circle after transform
            let vdot = dot(v, v);
            if (vdot <= 1.0) {
                // sample color (clamped to texture bounds)
                let sampleCoords: vec2<i32> = coords + vec2<i32>(x, y);
                let color: vec3<f32> = sampleTexture(sampleCoords, texSize_i);
                let saturatedColor: vec3<f32> = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));

                // polynomial weights for 8 sectors
                let polyWeights: array<f32, 8> = computePolynomialWeights(v, zeta, eta);

                // sum of poly weights
                var weightSum: f32 = 0.0;
                for (var k: i32 = 0; k < 8; k = k + 1) {
                    weightSum = weightSum + polyWeights[k];
                }

                // skip if total poly weight is zero (no contribution)
                if (weightSum <= EPS) {
                    continue;
                }

                // Gaussian falloff based on distance in transformed space
                // The original used exp(-3.125 * dot(v,v)); keep similar scaling
                let gaussian: f32 = exp(-3.125 * vdot);

                // accumulate into sector statistics with normalized polynomial weights
                for (var k: i32 = 0; k < 8; k = k + 1) {
                    let normalizedPoly: f32 = polyWeights[k] / weightSum;
                    let finalWeight: f32 = normalizedPoly * gaussian;

                    sectors[k].colorSum = sectors[k].colorSum + saturatedColor * finalWeight;
                    sectors[k].colorSqSum = sectors[k].colorSqSum + (saturatedColor * saturatedColor) * finalWeight;
                    sectors[k].weightSum = sectors[k].weightSum + finalWeight;
                }
            }
        }
    }

    // Compute sector means, variances, and selection weights; accumulate final color
    var finalColor: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var totalSelectionWeight: f32 = 0.0;

    for (var k: i32 = 0; k < 8; k = k + 1) {
        if (sectors[k].weightSum > EPS) {
            // mean color for sector k
            let sectorMean: vec3<f32> = sectors[k].colorSum / sectors[k].weightSum;

            // second moment E[X^2]
            let secondMoment: vec3<f32> = sectors[k].colorSqSum / sectors[k].weightSum;

            // variance per channel (E[X^2] - (E[X])^2)
            let variance: vec3<f32> = secondMoment - sectorMean * sectorMean;
            // ensure non-negative variances
            let variancePos: vec3<f32> = max(variance, vec3<f32>(0.0, 0.0, 0.0));

            // total variance (sum channels)
            let sigma2: f32 = variancePos.r + variancePos.g + variancePos.b;

            // compute selection weight (Unity-inspired formula)
            // Protect hardness non-negativity
            let hardnessPos: f32 = max(kuwaharaParams.hardness, 0.0);
            // keep the original 1000.0 scaling factor as you had; can be tuned
            let baseVal: f32 = hardnessPos * 1000.0 * sigma2;
            // clamp base to non-negative region; pow(0, exponent) is allowed
            let exponent: f32 = 0.5 * kuwaharaParams.sharpness;
            // protect against negative base due to numerical issues:
            let baseClamped: f32 = max(baseVal, 0.0);

            // compute selectionWeight robustly
            let selectionWeight: f32 = 1.0 / (1.0 + pow(baseClamped, exponent));

            finalColor = finalColor + sectorMean * selectionWeight;
            totalSelectionWeight = totalSelectionWeight + selectionWeight;
        }
    }

    // If no sectors contributed, fall back to original pixel color
    var result: vec3<f32>;
    if (totalSelectionWeight > EPS) {
        result = finalColor / totalSelectionWeight;
    } else {
        result = sampleTexture(coords, texSize_i);
    }

    let clampedResult: vec3<f32> = clamp(result, vec3<f32>(0.0), vec3<f32>(1.0));

    // Write out to storage texture
    textureStore(outputTexture, coords, vec4<f32>(clampedResult, 1.0));
}
