struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    // Create a full-screen quad using 4 vertices for triangle-strip
    var pos = array<vec2f, 4>(
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0),
        vec2f(-1.0,  1.0),
        vec2f( 1.0,  1.0)
    );

    var texcoord = array<vec2f, 4>(
        vec2f(0.0, 1.0),
        vec2f(1.0, 1.0),
        vec2f(0.0, 0.0),
        vec2f(1.0, 0.0)
    );

    var output: VertexOutput;
    output.position = vec4f(pos[vertexIndex], 0.0, 1.0);
    output.texcoord = texcoord[vertexIndex];
    return output;
}
