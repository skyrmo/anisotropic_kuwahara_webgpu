<template>
    <div class="image-editor">
        <div class="controls">
            <input
                ref="fileInput"
                type="file"
                accept="image/*"
                @change="handleFileSelect"
                style="display: none"
            />
            <button @click="selectImage" class="select-button">Select Image</button>

            <div v-if="error" class="error">
                {{ error }}
            </div>

            <div v-if="loading || processing" class="loading">
                {{ loading ? "Loading image..." : "Applying color quantization..." }}
            </div>
        </div>

        <div class="canvas-container">
            <canvas ref="canvas"></canvas>
            <div v-if="!imageLoaded" class="placeholder">
                Select an image to display
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";
import { ImageEditorService } from "../services/Editor.service";

const canvas = ref<HTMLCanvasElement>();
const fileInput = ref<HTMLInputElement>();
const error = ref<string>("");

const loading = ref<boolean>(false);
const processing = ref<boolean>(false);
const imageLoaded = ref<boolean>(false);

let imageEditor: ImageEditorService | null = null;
// let currentImageFile: File | null = null;

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
    // currentImageFile = file;

    try {
        if (!imageEditor) {
            throw new Error("WebGPU service not initialized");
        }

        await imageEditor.loadImage(file);

        imageLoaded.value = true;
    } catch (err) {
        error.value = err instanceof Error ? err.message : "Failed to load image";
        console.error("Image loading error:", err);
    } finally {
        loading.value = false;
    }
};
</script>

<style scoped></style>
