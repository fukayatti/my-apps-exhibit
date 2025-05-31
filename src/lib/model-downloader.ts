import { ModelFile } from "./types";

export class ModelDownloader {
  private baseUrl: string;
  private files: ModelFile[];
  private downloadedFiles: Record<string, ArrayBuffer> = {};

  constructor(baseUrl?: string, files?: ModelFile[]) {
    this.baseUrl =
      baseUrl ||
      "https://huggingface.co/fukayatti0/small100-quantized-int8/resolve/main/";
    this.files = files || [
      { name: "model.onnx", size: "~150MB" },
      { name: "vocab.json", size: "~3.5MB" },
      { name: "tokenizer_config.json", size: "~2KB" },
      { name: "config.json", size: "~1KB" },
    ];
  }

  async downloadFile(
    filename: string,
    onProgress?: (loaded: number, total: number) => void
  ): Promise<ArrayBuffer> {
    const url = this.baseUrl + filename;

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const contentLength = response.headers.get("content-length");
      const total = parseInt(contentLength || "0", 10);
      let loaded = 0;

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Failed to get response reader");
      }

      const chunks: Uint8Array[] = [];

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        chunks.push(value);
        loaded += value.length;

        if (onProgress && total) {
          onProgress(loaded, total);
        }
      }

      // Uint8Arrayを結合
      const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
      const result = new Uint8Array(totalLength);
      let offset = 0;
      for (const chunk of chunks) {
        result.set(chunk, offset);
        offset += chunk.length;
      }

      return result.buffer;
    } catch (error) {
      console.error(
        `Error downloading ${filename} from ${this.baseUrl + filename}:`,
        error
      );
      throw error;
    }
  }

  async downloadAllFiles(
    onOverallProgress: (
      progress: number,
      currentFile: string,
      fileProgress: number
    ) => void,
    onFileComplete: (filename: string, size: number) => void
  ): Promise<Record<string, ArrayBuffer>> {
    const totalFiles = this.files.length;
    let completedFiles = 0;

    for (const file of this.files) {
      try {
        const buffer = await this.downloadFile(file.name, (loaded, total) => {
          const fileProgress = loaded / total;
          const overallProgress = (completedFiles + fileProgress) / totalFiles;
          onOverallProgress(
            overallProgress * 100,
            file.name,
            fileProgress * 100
          );
        });

        this.downloadedFiles[file.name] = buffer;
        completedFiles++;
        onFileComplete(file.name, buffer.byteLength);
      } catch (error) {
        throw new Error(
          `${file.name} のダウンロードに失敗: ${(error as Error).message}`
        );
      }
    }

    return this.downloadedFiles;
  }

  getDownloadedFile(filename: string): ArrayBuffer | undefined {
    return this.downloadedFiles[filename];
  }

  getFiles(): ModelFile[] {
    return this.files;
  }
}
