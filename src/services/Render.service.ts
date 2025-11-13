import vertexShaderSource from "../shaders/vertex.wgsl?raw";
import fragmentShaderSource from "../shaders/fragment.wgsl?raw";
import { WebGPUCore } from "./WebGPU.service";

export class RenderService {
    private core: WebGPUCore;
    private pipeline: GPURenderPipeline | null = null;
    private sampler: GPUSampler | null = null;

    constructor() {
        this.core = WebGPUCore.getInstance();
    }

    async initialize(): Promise<void> {
        const device = this.core.getDevice();

        // Create sampler
        this.sampler = device.createSampler({
            magFilter: "nearest",
            minFilter: "nearest",
        });

        // Create render pipeline
        const vertexShaderModule = device.createShaderModule({
            code: vertexShaderSource,
        });
        // Create render pipeline
        const fragmentShaderModule = device.createShaderModule({
            code: fragmentShaderSource,
        });

        this.pipeline = device.createRenderPipeline({
            layout: "auto",
            vertex: {
                module: vertexShaderModule,
                entryPoint: "vertexMain",
            },
            fragment: {
                module: fragmentShaderModule,
                entryPoint: "fragmentMain",
                targets: [{ format: this.core.getCanvasFormat() }],
            },
            primitive: {
                topology: "triangle-strip",
            },
        });
    }

    render(texture: GPUTexture): void {
        const device = this.core.getDevice();
        const context = this.core.getContext();

        if (!context || !this.pipeline || !this.sampler) {
            console.error("Render service not properly initialized");
            return;
        }

        // Create bind group
        const bindGroup = device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.sampler },
                { binding: 1, resource: texture.createView() },
            ],
        });

        // Create command encoder
        const encoder = device.createCommandEncoder();

        // Begin render pass
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: context.getCurrentTexture().createView(),
                    clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
                    loadOp: "clear",
                    storeOp: "store",
                },
            ],
        });

        // Draw
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(4);
        pass.end();

        // Submit
        device.queue.submit([encoder.finish()]);
    }

    getSampler(): GPUSampler | null {
        return this.sampler;
    }
}
