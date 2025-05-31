import { ModelFile } from "./types";

export class ModelDownloader {
  private baseUrl: string;
  private files: ModelFile[];
  private downloadedFiles: Record<string, ArrayBuffer> = {};

  constructor(baseUrl?: string, files?: ModelFile[]) {
    this.baseUrl =
      baseUrl ||
      "https://huggingface.co/https://huggingface.co/alirezamsh/small100/tree/main/resolve/main/";
    this.files = files || [
      { name: "model.onnx", size: "~824MB" }, // https://huggingface.co/alirezamsh/small100/tree/main の model.onnx は約824MB
      { name: "vocab.json", size: "~3.5MB" },
      { name: "sentencepiece.bpe.model", size: "~4MB" }, // SentencePieceモデルファイル
      { name: "config.json", size: "~1KB" },
      { name: "tokenizer_config.json", size: "~2KB" },
    ];
  }

  async downloadFile(
    filename: string,
    onProgress?: (loaded: number, total: number) => void,
    maxRetries = 3,
    retryDelay = 1000
  ): Promise<ArrayBuffer> {
    const url = this.baseUrl + filename;
    let attempt = 0;

    while (attempt < maxRetries) {
      try {
        console.log(
          `[Downloader] Attempt ${
            attempt + 1
          }/${maxRetries} to download ${filename} from ${url}`
        );
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status} for ${url}`);
        }

        const contentLength = response.headers.get("content-length");
        const total = parseInt(contentLength || "0", 10);
        let loaded = 0;

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error("Failed to get response reader");
        }

        const chunks: Uint8Array[] = [];
        console.log(
          `[Downloader] Starting download for ${filename}, Total size: ${
            total > 0 ? (total / (1024 * 1024)).toFixed(2) + "MB" : "Unknown"
          }`
        );

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
        const totalLength = chunks.reduce(
          (acc, chunk) => acc + chunk.length,
          0
        );
        const result = new Uint8Array(totalLength);
        let offset = 0;
        for (const chunk of chunks) {
          result.set(chunk, offset);
          offset += chunk.length;
        }
        console.log(
          `[Downloader] Successfully downloaded ${filename} (${(
            totalLength /
            (1024 * 1024)
          ).toFixed(2)}MB)`
        );
        return result.buffer;
      } catch (error) {
        attempt++;
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        console.error(
          `[Downloader] Error downloading ${filename} (Attempt ${attempt}/${maxRetries}): ${errorMessage}. URL: ${url}`
        );
        if (attempt >= maxRetries) {
          throw new Error(
            `Failed to download ${filename} after ${maxRetries} attempts. Last error: ${errorMessage}`
          );
        }
        console.log(
          `[Downloader] Retrying download of ${filename} in ${
            retryDelay / 1000
          }s...`
        );
        await new Promise((resolve) => setTimeout(resolve, retryDelay));
      }
    }
    // Should not reach here, but to satisfy TypeScript compiler
    throw new Error(
      `Exhausted retries for ${filename} without success or throwing an error.`
    );
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
