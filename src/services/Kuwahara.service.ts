import { WebGPUCore } from "./WebGPU.service";
import { TextureManagerService } from "./Textures.service";
import type { KuwaharaParams } from "../types/types";

import structureTensorShaderCode from "../shaders/kuwahara/structure-tensor.wgsl?raw";
import blurShaderCode from "../shaders/kuwahara/blur.wgsl?raw";
import eigenvectorShaderCode from "../shaders/kuwahara/eigenvector.wgsl?raw";
import kuwaharaShaderCode from "../shaders/kuwahara/kuwahara.wgsl?raw";

export class KuwaharaService {
    private core: WebGPUCore;
    private textureManager: TextureManagerService;

    // Pipelines
    private structureTensorPipeline!: GPUComputePipeline;
    private blurHorizontalPipeline!: GPUComputePipeline;
    private blurVerticalPipeline!: GPUComputePipeline;
    private eigenvectorPipeline!: GPUComputePipeline;
    private kuwaharaPipeline!: GPUComputePipeline;

    // Buffers
    private kuwaharaParamsBuffer!: GPUBuffer;

    // Textures
    private structureTensorTexture!: GPUTexture;
    private blurHorizontalTexture!: GPUTexture;
    private blurVerticalTexture!: GPUTexture;
    private eigenvectorTexture!: GPUTexture;
    private kuwaharaOutputTexture!: GPUTexture;

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

        const eigenvectorShaderModule = device.createShaderModule({
            label: "Eigenvector shader module",
            code: eigenvectorShaderCode,
        });

        this.eigenvectorPipeline = device.createComputePipeline({
            layout: "auto",
            compute: {
                module: eigenvectorShaderModule,
                entryPoint: "computeMain",
            },
        });

        const kuwaharaShaderModule = device.createShaderModule({
            label: "Kuwahara shader module",
            code: kuwaharaShaderCode,
        });

        this.kuwaharaPipeline = device.createComputePipeline({
            layout: "auto",
            compute: {
                module: kuwaharaShaderModule,
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
            size: 32, // 7 parameters aligned to 16-byte boundary: 32 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Use proper zeta default calculation
        const effectiveZeta =
            this.kuwaharaParams.zeta || 2.0 / (this.kuwaharaParams.kernelSize / 2);

        // Create properly typed buffer data
        const bufferData = new ArrayBuffer(32);
        const intView = new Int32Array(bufferData);
        const floatView = new Float32Array(bufferData);

        // Write integer at correct position
        intView[0] = this.kuwaharaParams.kernelSize; // offset 0

        // Write floats at correct positions
        floatView[1] = this.kuwaharaParams.sharpness; // offset 4
        floatView[2] = this.kuwaharaParams.hardness; // offset 8
        floatView[3] = this.kuwaharaParams.alpha; // offset 12
        floatView[4] = this.kuwaharaParams.zeroCrossing; // offset 16
        floatView[5] = effectiveZeta; // offset 20
        floatView[6] = this.kuwaharaParams.sigma; // offset 24

        device.queue.writeBuffer(this.kuwaharaParamsBuffer, 0, bufferData);

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
        const device = this.core.getDevice();

        // Get input textures
        const originalTexture = this.textureManager.getTexture("original");
        const eigenvectorTexture = this.textureManager.getTexture("eigenvector_output");

        if (!originalTexture || !eigenvectorTexture) {
            throw new Error("Required textures not found for Kuwahara filtering");
        }

        // Create output texture
        this.kuwaharaOutputTexture = device.createTexture({
            size: { width: originalTexture.width, height: originalTexture.height },
            format: "rgba8unorm",
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
        });

        // Execute Kuwahara filter
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.kuwaharaPipeline);

        pass.setBindGroup(
            0,
            device.createBindGroup({
                layout: this.kuwaharaPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: originalTexture.texture.createView() },
                    { binding: 1, resource: eigenvectorTexture.texture.createView() },
                    { binding: 2, resource: this.kuwaharaOutputTexture.createView() },
                    { binding: 3, resource: { buffer: this.kuwaharaParamsBuffer } },
                ],
            }),
        );

        const workgroupsX = Math.ceil(originalTexture.width / 8);
        const workgroupsY = Math.ceil(originalTexture.height / 8);
        pass.dispatchWorkgroups(workgroupsX, workgroupsY);
        pass.end();

        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // Store the final result
        this.textureManager.setTexture("kuwahara_output", {
            texture: this.kuwaharaOutputTexture,
            width: originalTexture.width,
            height: originalTexture.height,
            format: "rgba8unorm",
        });
    }
}
