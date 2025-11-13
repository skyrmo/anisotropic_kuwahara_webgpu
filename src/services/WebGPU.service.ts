export class WebGPUCore {
    private static instance: WebGPUCore | null = null;

    private device: GPUDevice | null = null;
    private adapter: GPUAdapter | null = null;
    private canvas: HTMLCanvasElement | null = null;
    private context: GPUCanvasContext | null = null;
    private canvasFormat: GPUTextureFormat = "bgra8unorm";

    private constructor() {}

    static getInstance(): WebGPUCore {
        if (!WebGPUCore.instance) {
            WebGPUCore.instance = new WebGPUCore();
        }
        return WebGPUCore.instance;
    }

    async initialize(canvas: HTMLCanvasElement): Promise<void> {
        if (this.device) {
            console.warn("WebGPU already initialized");
            return;
        }

        this.canvas = canvas;

        if (!navigator.gpu) {
            throw new Error("WebGPU is not supported in this browser");
        }

        this.adapter = await navigator.gpu.requestAdapter();
        if (!this.adapter) {
            throw new Error("Failed to get GPU adapter");
        }

        this.device = await this.adapter.requestDevice();

        this.context = canvas.getContext("webgpu");
        if (!this.context) {
            throw new Error("Failed to get WebGPU context");
        }

        this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();

        this.configureContext();
    }

    configureContext(width?: number, height?: number): void {
        if (!this.context || !this.device || !this.canvas) return;

        if (width && height) {
            this.canvas.width = width;
            this.canvas.height = height;
        }

        this.context.configure({
            device: this.device,
            format: this.canvasFormat,
            alphaMode: "premultiplied",
        });
    }

    getDevice(): GPUDevice {
        if (!this.device) {
            throw new Error("WebGPU not initialized. Call initialize() first.");
        }
        return this.device;
    }

    getContext(): GPUCanvasContext | null {
        return this.context;
    }

    getCanvasFormat(): GPUTextureFormat {
        return this.canvasFormat;
    }

    getCanvasDimensions(): { width: number; height: number } {
        if (!this.canvas) {
            return { width: 0, height: 0 };
        }
        return {
            width: this.canvas.width,
            height: this.canvas.height,
        };
    }

    isInitialized(): boolean {
        return this.device !== null;
    }

    destroy(): void {
        this.device = null;
        this.context = null;
        this.canvas = null;
        this.adapter = null;
        WebGPUCore.instance = null;
    }
}
