import { WebGPUCore } from "./WebGPU.service";
import type { ManagedTexture } from "../types/types";

export class TextureManagerService {
    private core: WebGPUCore;
    private textures: Map<string, ManagedTexture> = new Map();

    constructor() {
        this.core = WebGPUCore.getInstance();
    }

    async createImageTexture(
        imageFile: File | Blob,
        key: string,
    ): Promise<ManagedTexture> {
        const device = this.core.getDevice();
        const bitmap = await createImageBitmap(imageFile);

        const texture = device.createTexture({
            size: [bitmap.width, bitmap.height, 1],
            format: "rgba8unorm",
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.COPY_SRC |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });

        device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [
            bitmap.width,
            bitmap.height,
        ]);

        const managedTexture: ManagedTexture = {
            texture,
            width: bitmap.width,
            height: bitmap.height,
            format: "rgba8unorm",
        };

        // Store with key
        this.textures.set(key, managedTexture);

        return managedTexture;
    }

    getTexture(key: string): ManagedTexture | undefined {
        return this.textures.get(key);
    }

    setTexture(key: string, texture: ManagedTexture): void {
        this.textures.set(key, texture);
    }

    destroyTexture(key: string): void {
        const managed = this.textures.get(key);
        if (managed) {
            managed.texture.destroy();
            this.textures.delete(key);
        }
    }

    destroyAll(): void {
        for (const managed of this.textures.values()) {
            managed.texture.destroy();
        }
        this.textures.clear();
    }

    // async readTextureData(key: string): Promise<Uint8Array | null> {
    //     const managed = this.textures.get(key);
    //     if (!managed) {
    //         console.error(`Texture with key "${key}" not found`);
    //         return null;
    //     }

    //     const device = this.core.getDevice();

    //     // Calculate buffer size (4 bytes per pixel for RGBA)
    //     const bytesPerRow = Math.ceil((managed.width * 4) / 256) * 256; // Align to 256 bytes
    //     const bufferSize = bytesPerRow * managed.height;

    //     // Create a buffer to copy the texture data into
    //     const buffer = device.createBuffer({
    //         size: bufferSize,
    //         usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    //     });

    //     // Create command encoder
    //     const encoder = device.createCommandEncoder();

    //     // Copy texture to buffer
    //     encoder.copyTextureToBuffer(
    //         {
    //             texture: managed.texture,
    //             mipLevel: 0,
    //             origin: { x: 0, y: 0, z: 0 },
    //         },
    //         {
    //             buffer: buffer,
    //             offset: 0,
    //             bytesPerRow: bytesPerRow,
    //             rowsPerImage: managed.height,
    //         },
    //         {
    //             width: managed.width,
    //             height: managed.height,
    //             depthOrArrayLayers: 1,
    //         },
    //     );

    //     // Submit commands
    //     device.queue.submit([encoder.finish()]);

    //     // Wait for GPU to finish
    //     await device.queue.onSubmittedWorkDone();

    //     // Map the buffer for reading
    //     await buffer.mapAsync(GPUMapMode.READ);

    //     // Get the mapped data
    //     const mappedRange = buffer.getMappedRange();

    //     // Create a properly sized output array (remove padding)
    //     const outputData = new Uint8Array(managed.width * managed.height * 4);
    //     const tempData = new Uint8Array(mappedRange);

    //     // Copy row by row to remove padding
    //     for (let y = 0; y < managed.height; y++) {
    //         const srcOffset = y * bytesPerRow;
    //         const dstOffset = y * managed.width * 4;
    //         outputData.set(tempData.slice(srcOffset, srcOffset + managed.width * 4), dstOffset);
    //     }

    //     // Unmap the buffer
    //     buffer.unmap();

    //     // Destroy the buffer
    //     buffer.destroy();

    //     return outputData;
    // }
}
