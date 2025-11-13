import { WebGPUCore } from "./WebGPU.service";
import { RenderService } from "./Render.service";
import { TextureManagerService } from "./Textures.service";

export class ImageEditorService {
    private webGPUCore: WebGPUCore;
    private renderer: RenderService;
    private textureManager: TextureManagerService;

    constructor() {
        this.webGPUCore = WebGPUCore.getInstance();
        this.renderer = new RenderService();
        this.textureManager = new TextureManagerService();
    }

    async initialize(canvas: HTMLCanvasElement): Promise<void> {
        await this.webGPUCore.initialize(canvas);
        await this.renderer.initialize();
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

        this.render();
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
