import { FileStatus } from "./types";

export class ModelStorage {
  private models: Map<
    string,
    { buffer: ArrayBuffer; timestamp: number; size: number }
  > = new Map();
  private initialized = false;

  async init(): Promise<void> {
    if (!this.initialized) {
      this.models = new Map();
      this.initialized = true;
      console.log("[ModelStorage] In-memory storage initialized.");
    }
    return Promise.resolve();
  }

  async saveModel(
    filename: string,
    buffer: ArrayBuffer
    // metadata: Record<string, unknown> = {} // metadata は現状未使用のためコメントアウト
  ): Promise<string> {
    // IDBValidKey から string に変更 (ファイル名を返す)
    if (!this.initialized) await this.init();

    this.models.set(filename, {
      buffer: buffer,
      timestamp: Date.now(),
      size: buffer.byteLength,
      // ...metadata, // metadata は現状未使用
    });
    console.log(
      `[ModelStorage] Saved ${filename} to memory. Size: ${buffer.byteLength}`
    );
    return Promise.resolve(filename);
  }

  async getModel(filename: string): Promise<ArrayBuffer | null> {
    if (!this.initialized) await this.init();

    const modelData = this.models.get(filename);
    if (modelData) {
      console.log(`[ModelStorage] Retrieved ${filename} from memory.`);
      return Promise.resolve(modelData.buffer);
    }
    console.log(`[ModelStorage] ${filename} not found in memory.`);
    return Promise.resolve(null);
  }

  async hasModel(filename: string): Promise<boolean> {
    if (!this.initialized) await this.init();
    const exists = this.models.has(filename);
    console.log(
      `[ModelStorage] Check if ${filename} exists in memory: ${exists}`
    );
    return Promise.resolve(exists);
  }

  async getModelInfo(filename: string): Promise<FileStatus | null> {
    if (!this.initialized) await this.init();

    const modelData = this.models.get(filename);
    if (modelData) {
      const fileStatus: FileStatus = {
        filename: filename,
        size: modelData.size,
        timestamp: modelData.timestamp,
      };
      console.log(
        `[ModelStorage] Retrieved info for ${filename} from memory:`,
        fileStatus
      );
      return Promise.resolve(fileStatus);
    }
    console.log(`[ModelStorage] Info for ${filename} not found in memory.`);
    return Promise.resolve(null);
  }

  // getAllModelInfo は getAllFilesInfo と機能が重複するため削除
  // async getAllModelInfo(): Promise<FileStatus[]> {
  //   // ...
  // }

  async getAllFilesInfo(): Promise<FileStatus[]> {
    if (!this.initialized) await this.init();

    const filesInfo: FileStatus[] = [];
    for (const [filename, data] of this.models.entries()) {
      filesInfo.push({
        filename: filename,
        size: data.size,
        timestamp: data.timestamp,
      });
    }
    console.log(
      "[ModelStorage] Retrieved info for all files in memory:",
      filesInfo
    );
    return Promise.resolve(filesInfo);
  }

  async deleteModel(filename: string): Promise<boolean> {
    if (!this.initialized) await this.init();

    const deleted = this.models.delete(filename);
    if (deleted) {
      console.log(`[ModelStorage] Deleted ${filename} from memory.`);
    } else {
      console.log(
        `[ModelStorage] ${filename} not found in memory, nothing to delete.`
      );
    }
    return Promise.resolve(deleted);
  }

  async clearAll(): Promise<boolean> {
    if (!this.initialized) await this.init();

    this.models.clear();
    console.log("[ModelStorage] Cleared all models from memory.");
    return Promise.resolve(true);
  }
}
