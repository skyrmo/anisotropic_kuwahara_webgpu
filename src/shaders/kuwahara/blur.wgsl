@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> kuwaharaParams: KuwaharaParams;

struct KuwaharaParams {
    kernelSize: i32,
    sharpness: f32,
    hardness: f32,
    alpha: f32,
    zeroCrossing: f32,
    zeta: f32,
    sigma: f32,
}


fn gaussianWeight(x: f32, sigma: f32) -> f32 {
    let sigmaSq = sigma * sigma;
    return exp(-(x * x) / (2.0 * sigmaSq));
}

fn horizontalBlur(coords: vec2i, texSize: vec2i, sigma: f32, radius: i32) -> vec4f {
    var sum = vec4f(0.0);
    var weightSum = 0.0;

    // Sample along x-axis
    for (var dx: i32 = -radius; dx <= radius; dx++) {
        let sampleCoords = vec2i(
            clamp(coords.x + dx, 0, i32(texSize.x) - 1),
            coords.y
        );

        let weight = gaussianWeight(f32(dx), sigma);
        let sample = textureLoad(inputTexture, sampleCoords, 0);

        sum += sample * weight;
        weightSum += weight;
    }

    return sum / weightSum;
}

fn verticalBlur(coords: vec2i, texSize: vec2i, sigma: f32, radius: i32) -> vec4f {
    var sum = vec4f(0.0);
    var weightSum = 0.0;

    // Sample along y-axis
    for (var dy: i32 = -radius; dy <= radius; dy++) {
        let sampleCoords = vec2i(
            coords.x,
            clamp(coords.y + dy, 0, i32(texSize.y) - 1)
        );

        let weight = gaussianWeight(f32(dy), sigma);
        let sample = textureLoad(inputTexture, sampleCoords, 0);

        sum += sample * weight;
        weightSum += weight;
    }

    return sum / weightSum;
}

@compute @workgroup_size(8, 8)
fn horizontalMain(@builtin(global_invocation_id) globalId: vec3u) {
    let texSize = textureDimensions(inputTexture);
    let coords = vec2i(globalId.xy);

    if (coords.x >= i32(texSize.x) || coords.y >= i32(texSize.y)) {
        return;
    }

    let result = horizontalBlur(coords, vec2i(texSize), kuwaharaParams.sigma, kuwaharaParams.kernelSize);
    textureStore(outputTexture, coords, result);
}

@compute @workgroup_size(8, 8)
fn verticalMain(@builtin(global_invocation_id) globalId: vec3u) {
    let texSize = textureDimensions(inputTexture);
    let coords = vec2i(globalId.xy);

    if (coords.x >= i32(texSize.x) || coords.y >= i32(texSize.y)) {
        return;
    }

    let result = verticalBlur(coords, vec2i(texSize), kuwaharaParams.sigma, kuwaharaParams.kernelSize);
    textureStore(outputTexture, coords, result);
}
