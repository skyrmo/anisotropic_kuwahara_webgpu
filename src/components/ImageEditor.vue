<template>
    <div class="image-editor">
        <div class="controls-section">
            <input
                ref="fileInput"
                type="file"
                accept="image/*"
                @change="handleFileSelect"
                style="display: none"
            />
            <button @click="selectImage" class="select-button">Select Image</button>

            <!-- Kuwahara parameter controls -->
            <div v-if="imageLoaded" class="shader-controls">
                <h3>Kuwahara Filter Parameters</h3>

                <div
                    class="slider-group"
                    v-for="(value, key) in kuwaharaParams"
                    :key="key"
                >
                    <label :for="key">{{ key }}: {{ value }}</label>
                    <input
                        type="range"
                        :id="key"
                        :min="sliderRanges[key].min"
                        :max="sliderRanges[key].max"
                        :step="sliderRanges[key].step"
                        v-model.number="kuwaharaParams[key]"
                    />
                </div>
            </div>

            <div v-if="error" class="error">
                {{ error }}
            </div>

            <div v-if="loading || processing" class="loading">
                {{ loading ? "Loading image..." : "Applying Kuwahara filter..." }}
            </div>
        </div>

        <div class="canvas-section">
            <div class="canvas-wrapper">
                <canvas ref="canvas"></canvas>
                <div v-if="!imageLoaded" class="placeholder">
                    Select an image to display
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from "vue";
import { ImageEditorService } from "../services/Editor.service";
import type { KuwaharaParams } from "../types/types";

const canvas = ref<HTMLCanvasElement>();
const fileInput = ref<HTMLInputElement>();
const error = ref<string>("");

const loading = ref<boolean>(false);
const processing = ref<boolean>(false);
const imageLoaded = ref<boolean>(false);

let imageEditor: ImageEditorService | null = null;

// --- Kuwahara Params and slider settings ---
const kuwaharaParams = ref<KuwaharaParams>({
    kernelSize: 10,
    sharpness: 0.8,
    hardness: 0.5,
    alpha: 4.0,
    zeroCrossing: 0.1,
    zeta: 0.05,
    sigma: 5.0,
});

const sliderRanges: Record<
    keyof KuwaharaParams,
    { min: number; max: number; step: number }
> = {
    kernelSize: { min: 1, max: 160, step: 1 },
    sharpness: { min: 0.1, max: 20.0, step: 0.1 },
    hardness: { min: 0.1, max: 20.0, step: 0.1 },
    alpha: { min: 0.01, max: 10.0, step: 0.01 },
    zeroCrossing: { min: 0, max: 3.14, step: 0.1 },
    zeta: { min: 0.01, max: 6.0, step: 0.01 },
    sigma: { min: 0.5, max: 200.0, step: 0.1 },
};

// --- Lifecycle ---
onMounted(async () => {
    if (!canvas.value) return;

    try {
        imageEditor = new ImageEditorService();
        await imageEditor.initialize(canvas.value);
    } catch (err) {
        error.value =
            err instanceof Error ? err.message : "Failed to initialize WebGPU";
        console.error("WebGPU initialization error:", err);
    }
});

onUnmounted(() => {
    if (imageEditor) {
        imageEditor.destroy();
    }
});

const selectImage = () => {
    fileInput.value?.click();
};

const handleFileSelect = async (event: Event) => {
    const target = event.target as HTMLInputElement;
    const file = target.files?.[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
        error.value = "Please select a valid image file";
        return;
    }

    error.value = "";
    loading.value = true;

    try {
        if (!imageEditor) throw new Error("WebGPU service not initialized");

        await imageEditor.loadImage(file);
        imageLoaded.value = true;

        // Apply shader initially
        await applyKuwaharaFilter();
    } catch (err) {
        error.value = err instanceof Error ? err.message : "Failed to load image";
        console.error("Image loading error:", err);
    } finally {
        loading.value = false;
    }
};

watch(
    kuwaharaParams,
    async () => {
        if (imageEditor && imageLoaded.value) {
            await applyKuwaharaFilter();
        }
    },
    { deep: true },
);

const applyKuwaharaFilter = async () => {
    if (!imageEditor) return;

    processing.value = true;
    try {
        await imageEditor.runKuwaharaFilter({ ...kuwaharaParams.value });
    } catch (err) {
        console.error("Shader apply error:", err);
        error.value =
            err instanceof Error ? err.message : "Failed to apply Kuwahara filter";
    } finally {
        processing.value = false;
    }
};
</script>

<style scoped>
.image-editor {
    display: flex;
    height: 100vh;
    width: stretch;
    gap: 1rem;
    justify-content: space-between;
}

.controls-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
}

.select-button {
    margin: 1rem 4rem;
}

.shader-controls {
    max-width: 300px;
    border-radius: 8px;
    padding: 1rem;
}

.slider-group {
    display: flex;
    flex-direction: column;
    margin-bottom: 0.5rem;
}

.slider-group label {
    font-size: 0.9rem;
    margin-bottom: 0.25rem;
}

.canvas-section {
    flex: 1 0 auto;
    display: flex;
}
.canvas {
    max-width: 100%;
}
.canvas-wrapper {
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
}

.placeholder {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #777;
}
</style>
