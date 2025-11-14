import { WebGPUCore } from "./WebGPU.service";
import { RenderService } from "./Render.service";
import { TextureManagerService } from "./Textures.service";
import { KuwaharaService } from "./Kuwahara.service";
import type { KuwaharaParams } from "../types/types";

export class ImageEditorService {
    private webGPUCore: WebGPUCore;
    private renderer: RenderService;
    private textureManager: TextureManagerService;
    private kuwahara: KuwaharaService;

    constructor() {
        this.webGPUCore = WebGPUCore.getInstance();
        this.renderer = new RenderService();
        this.textureManager = new TextureManagerService();
        this.kuwahara = new KuwaharaService(this.textureManager);
    }

    async initialize(canvas: HTMLCanvasElement): Promise<void> {
        await this.webGPUCore.initialize(canvas);
        await this.renderer.initialize();
        await this.kuwahara.initialize();
    }

    async loadImage(imageFile: File | Blob): Promise<void> {
        // Clear previous textures
        this.textureManager.destroyAll();

        // create new manage texztureand store it wih the key "original"
        const managedTexture = await this.textureManager.createImageTexture(
            imageFile,
            "original",
        );

        // Update canvas size
        this.webGPUCore.configureContext(managedTexture.width, managedTexture.height);
    }

    async runKuwaharaFilter(kuwaharaParams: KuwaharaParams) {
        await this.kuwahara.runKuwahara(kuwaharaParams);

        this.render("kuwahara_output");
    }

    private render(textureKey = "original"): void {
        // retrieve slected texture, based off provided key
        const managed = this.textureManager.getTexture(textureKey);

        if (managed) {
            this.renderer.render(managed.texture);
        } else {
            console.error(`No texture found with key: ${textureKey}`);
        }
    }

    destroy(): void {
        this.textureManager.destroyAll();
        this.webGPUCore.destroy();
    }
}
