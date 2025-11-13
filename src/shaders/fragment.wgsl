struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
};

@group(0) @binding(0) var mySampler: sampler;
@group(0) @binding(1) var myTexture: texture_2d<f32>;

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    let color = textureSample(myTexture, mySampler, input.texcoord);

    // Normal operation
    return color;
}
