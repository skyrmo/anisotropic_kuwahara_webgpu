import { WebGPUCore } from "./WebGPU.service";
import { TextureManagerService } from "./Textures.service";
import type { KuwaharaParams } from "../types/types";

import structureTensorShaderCode from "../shaders/kuwahara/structure-tensor.wgsl?raw";
import blurShaderCode from "../shaders/kuwahara/blur.wgsl?raw";
import eigenvectorShaderCode from "../shaders/kuwahara/eigenvector.wgsl?raw";

export class KuwaharaService {
    private core: WebGPUCore;
    private textureManager: TextureManagerService;

    // Pipelines
    private structureTensorPipeline!: GPUComputePipeline;
    private blurHorizontalPipeline!: GPUComputePipeline;
    private blurVerticalPipeline!: GPUComputePipeline;
    private eigenvectorPipeline!: GPUComputePipeline;

    // Buffers
    private kuwaharaParamsBuffer!: GPUBuffer;

    // Textures
    private structureTensorTexture!: GPUTexture;
    private blurHorizontalTexture!: GPUTexture;
    private blurVerticalTexture!: GPUTexture;
    private eigenvectorTexture!: GPUTexture;

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
            label: "Struc Tensor shader module",
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

        // create pipelines
        const blurShaderModule = device.createShaderModule({
            label: "Blur shader module",
            code: blurShaderCode,
        });

        this.blurHorizontalPipeline = device.createComputePipeline({
            layout: "auto",
            compute: {
                module: blurShaderModule,
                entryPoint: "horizontalMain",
            },
        });

        this.blurVerticalPipeline = device.createComputePipeline({
            layout: "auto",
            compute: {
                module: blurShaderModule,
                entryPoint: "verticalMain",
            },
        });

        // create pipelines
        const eigenvectorShaderModule = device.createShaderModule({
            label: "Eigenvector shader module",
            code: eigenvectorShaderCode,
        });

        // Create compute pipelines
        this.eigenvectorPipeline = device.createComputePipeline({
            layout: "auto",
            compute: {
                module: eigenvectorShaderModule,
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

    private async runStructureTensorSmoothing(): Promise<void> {
        await this.gaussianBlurPass(true); // horizontal
        await this.gaussianBlurPass(false); // vertical
    }

    async gaussianBlurPass(isHorizontal: boolean) {
        const device = this.core.getDevice();

        // Get input texture
        const inputTexture = isHorizontal
            ? this.textureManager.getTexture("structure_tensor")
            : this.textureManager.getTexture("horizontal_blur");

        if (!inputTexture) {
            throw new Error(
                `Input texture not found for ${isHorizontal ? "horizontal" : "vertical"} blur`,
            );
        }

        // Create output texture
        const outputTexture = device.createTexture({
            size: { width: inputTexture.width, height: inputTexture.height },
            format: "rgba16float",
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
        });

        // Create uniform buffer for kuwahara parameters
        this.kuwaharaParamsBuffer = device.createBuffer({
            size: 32, // 8 floats * 4 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(
            this.kuwaharaParamsBuffer,
            0,
            new Uint32Array([
                this.kuwaharaParams.kernelSize,
                this.kuwaharaParams.sharpness,
                this.kuwaharaParams.hardness,
                this.kuwaharaParams.alpha,
                this.kuwaharaParams.zeroCrossing,
                this.kuwaharaParams.zeta,
                this.kuwaharaParams.numSectors,
                this.kuwaharaParams.numPasses,
            ]),
        );

        // Execute blur pass with appropriate pipeline
        const pipeline = isHorizontal
            ? this.blurHorizontalPipeline
            : this.blurVerticalPipeline;

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);

        pass.setBindGroup(
            0,
            device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: inputTexture.texture.createView() },
                    { binding: 1, resource: outputTexture.createView() },
                    { binding: 2, resource: { buffer: this.kuwaharaParamsBuffer } },
                ],
            }),
        );

        const workgroupsX = Math.ceil(inputTexture.width / 8);
        const workgroupsY = Math.ceil(inputTexture.height / 8);
        pass.dispatchWorkgroups(workgroupsX, workgroupsY);
        pass.end();

        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // Store output
        const textureKey = isHorizontal ? "horizontal_blur" : "blur_output";
        if (isHorizontal) {
            this.blurHorizontalTexture = outputTexture;
        } else {
            this.blurVerticalTexture = outputTexture;
        }

        this.textureManager.setTexture(textureKey, {
            texture: outputTexture,
            width: inputTexture.width,
            height: inputTexture.height,
            format: "rgba16float",
        });
    }

    private async runEigenvectorAnalysis(): Promise<void> {
        const device = this.core.getDevice();

        // Get input texture
        const inputTexture = this.textureManager.getTexture("blur_output");

        if (!inputTexture) {
            throw new Error(`Input texture not found for blur output`);
        }

        // Create output texture
        const outputTexture = device.createTexture({
            size: { width: inputTexture.width, height: inputTexture.height },
            format: "rgba16float",
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
        });

        // Execute blur pass with appropriate pipeline
        const pipeline = this.eigenvectorPipeline;

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);

        pass.setBindGroup(
            0,
            device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: inputTexture.texture.createView() },
                    { binding: 1, resource: outputTexture.createView() },
                ],
            }),
        );

        const workgroupsX = Math.ceil(inputTexture.width / 8);
        const workgroupsY = Math.ceil(inputTexture.height / 8);
        pass.dispatchWorkgroups(workgroupsX, workgroupsY);
        pass.end();

        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        this.textureManager.setTexture("eigenvector_output", {
            texture: outputTexture,
            width: inputTexture.width,
            height: inputTexture.height,
            format: "rgba16float",
        });
    }

    private async runKuwaharaFiltering(): Promise<void> {
        console.log(this.kuwaharaParams);
    }
}
