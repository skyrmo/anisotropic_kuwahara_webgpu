import { WebGPUCore } from "./WebGPU.service";
import { TextureManagerService } from "./Textures.service";
import type { KuwaharaParams } from "../types/types";

import structureTensorShaderCode from "../shaders/kuwahara/structure-tensor.wgsl?raw";
// import updateShaderCode from "../shaders/slic/update.wgsl?raw";
// import outputShaderCode from "../shaders/slic/output.wgsl?raw";

export class KuwaharaService {
    private core: WebGPUCore;
    private textureManager: TextureManagerService;

    // Pipelines
    private structureTensorPipeline!: GPUComputePipeline;

    // Buffers

    // Textures
    private structureTensorTexture!: GPUTexture;

    // Parameters
    private kuwaharaParams!: KuwaharaParams;

    constructor(textureManager: TextureManagerService) {
        this.core = WebGPUCore.getInstance();
        this.textureManager = textureManager;
    }

    async initialize() {
        const device = this.core.getDevice();

        // create pipelines
        const structureTensorShaderModule = device.createShaderModule({
            label: "SLIC init centroids shader",
            code: structureTensorShaderCode,
        });

        // Create compute pipelines
        this.structureTensorPipeline = device.createComputePipeline({
            layout: "auto",
            compute: {
                module: structureTensorShaderModule,
                entryPoint: "computeMain",
            },
        });
    }

    async runKuwahara(kuwaharaParams: KuwaharaParams): Promise<void> {
        this.kuwaharaParams = kuwaharaParams;

        // 1. Structure Tensor Computation
        await this.runStructureTensorComputation();

        // 2. Structure Tensor Smoothing
        await this.runStructureTensorSmoothing();

        // 3. Eigenvector Analysis
        await this.runEigenvectorAnalysis();

        // 4. Anisotropic Kuwahara Filtering
        await this.runKuwaharaFiltering();
    }

    private async runStructureTensorComputation(): Promise<void> {
        const device = this.core.getDevice();

        // Get input texture
        const inputTexture = this.textureManager.getTexture("original");
        if (!inputTexture) {
            throw new Error(`Texture with key "original" not found`);
        }

        // Labels texture: stores which centroid each pixel belongs to
        this.structureTensorTexture = device.createTexture({
            size: { width: inputTexture.width, height: inputTexture.height },
            format: "rgba16float",
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.structureTensorPipeline);
        pass.setBindGroup(
            0,
            device.createBindGroup({
                layout: this.structureTensorPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: inputTexture.texture.createView() },
                    { binding: 1, resource: this.structureTensorTexture.createView() },
                ],
            }),
        );

        const workgroupsX = Math.ceil(inputTexture.width / 8);
        const workgroupsY = Math.ceil(inputTexture.height / 8);
        pass.dispatchWorkgroups(workgroupsX, workgroupsY);
        pass.end();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // Store the visualization texture for rendering
        this.textureManager.setTexture("structure_tensor", {
            texture: this.structureTensorTexture,
            width: inputTexture.width,
            height: inputTexture.height,
            format: "rgba16float",
        });
    }

    private async runStructureTensorSmoothing(): Promise<void> {}

    private async runEigenvectorAnalysis(): Promise<void> {}

    private async runKuwaharaFiltering(): Promise<void> {
        console.log(this.kuwaharaParams);
    }
}
