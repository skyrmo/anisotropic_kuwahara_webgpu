export interface ManagedTexture {
    texture: GPUTexture;
    width: number;
    height: number;
    format: GPUTextureFormat;
}

export interface KuwaharaParams {
    kernelSize: number;
    sharpness: number;
    hardness: number;
    alpha: number;
    zeroCrossing: number;
    zeta: number;
    sigma: number;
}
