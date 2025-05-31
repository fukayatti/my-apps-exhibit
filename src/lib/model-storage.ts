import { FileStatus } from "./types";

export class ModelStorage {
  private dbName = "ModelStorage";
  private version = 1;
  private storeName = "models";
  private db: IDBDatabase | null = null;
  private initialized = false;

  async init(): Promise<void> {
    if (this.initialized && this.db) {
      console.log("[ModelStorage] IndexedDB already initialized.");
      return;
    }

    console.log("[ModelStorage] Initializing IndexedDB...");
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onerror = () => {
        console.error("[ModelStorage] IndexedDB開始エラー:", request.error);
        reject(new Error(`IndexedDBの初期化に失敗: ${request.error}`));
      };

      request.onsuccess = () => {
        this.db = request.result;
        this.initialized = true;
        console.log(
          "[ModelStorage] ✓ IndexedDB successfully initialized with database:",
          this.dbName
        );
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // 既存のストアがあれば削除
        if (db.objectStoreNames.contains(this.storeName)) {
          db.deleteObjectStore(this.storeName);
        }

        // 新しいオブジェクトストアを作成
        const store = db.createObjectStore(this.storeName, {
          keyPath: "filename",
        });
        store.createIndex("timestamp", "timestamp", { unique: false });
        console.log("[ModelStorage] IndexedDB object store created.");
      };
    });
  }

  async saveModel(filename: string, buffer: ArrayBuffer): Promise<string> {
    if (!this.initialized || !this.db) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readwrite");
      const store = transaction.objectStore(this.storeName);

      const modelData = {
        filename: filename,
        buffer: buffer,
        timestamp: Date.now(),
        size: buffer.byteLength,
      };

      const request = store.put(modelData);

      request.onsuccess = () => {
        console.log(
          `[ModelStorage] Saved ${filename} to IndexedDB. Size: ${buffer.byteLength}`
        );
        resolve(filename);
      };

      request.onerror = () => {
        console.error(
          `[ModelStorage] Error saving ${filename}:`,
          request.error
        );
        reject(new Error(`Failed to save ${filename}: ${request.error}`));
      };
    });
  }

  async getModel(filename: string): Promise<ArrayBuffer | null> {
    if (!this.initialized || !this.db) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readonly");
      const store = transaction.objectStore(this.storeName);
      const request = store.get(filename);

      request.onsuccess = () => {
        if (request.result) {
          console.log(`[ModelStorage] Retrieved ${filename} from IndexedDB.`);
          resolve(request.result.buffer);
        } else {
          console.log(`[ModelStorage] ${filename} not found in IndexedDB.`);
          resolve(null);
        }
      };

      request.onerror = () => {
        console.error(
          `[ModelStorage] Error retrieving ${filename}:`,
          request.error
        );
        reject(new Error(`Failed to retrieve ${filename}: ${request.error}`));
      };
    });
  }

  async hasModel(filename: string): Promise<boolean> {
    if (!this.initialized || !this.db) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readonly");
      const store = transaction.objectStore(this.storeName);
      const request = store.count(filename);

      request.onsuccess = () => {
        const exists = request.result > 0;
        console.log(
          `[ModelStorage] Check if ${filename} exists in IndexedDB: ${exists}`
        );
        resolve(exists);
      };

      request.onerror = () => {
        console.error(
          `[ModelStorage] Error checking ${filename}:`,
          request.error
        );
        reject(new Error(`Failed to check ${filename}: ${request.error}`));
      };
    });
  }

  async getModelInfo(filename: string): Promise<FileStatus | null> {
    if (!this.initialized || !this.db) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readonly");
      const store = transaction.objectStore(this.storeName);
      const request = store.get(filename);

      request.onsuccess = () => {
        if (request.result) {
          const fileStatus: FileStatus = {
            filename: request.result.filename,
            size: request.result.size,
            timestamp: request.result.timestamp,
          };
          console.log(
            `[ModelStorage] Retrieved info for ${filename} from IndexedDB:`,
            fileStatus
          );
          resolve(fileStatus);
        } else {
          console.log(
            `[ModelStorage] Info for ${filename} not found in IndexedDB.`
          );
          resolve(null);
        }
      };

      request.onerror = () => {
        console.error(
          `[ModelStorage] Error getting info for ${filename}:`,
          request.error
        );
        reject(
          new Error(`Failed to get info for ${filename}: ${request.error}`)
        );
      };
    });
  }

  async getAllFilesInfo(): Promise<FileStatus[]> {
    if (!this.initialized || !this.db) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readonly");
      const store = transaction.objectStore(this.storeName);
      const request = store.getAll();

      request.onsuccess = () => {
        const filesInfo: FileStatus[] = request.result.map((item) => ({
          filename: item.filename,
          size: item.size,
          timestamp: item.timestamp,
        }));
        console.log(
          "[ModelStorage] Retrieved info for all files in IndexedDB:",
          filesInfo
        );
        resolve(filesInfo);
      };

      request.onerror = () => {
        console.error(
          "[ModelStorage] Error getting all files info:",
          request.error
        );
        reject(new Error(`Failed to get all files info: ${request.error}`));
      };
    });
  }

  async deleteModel(filename: string): Promise<boolean> {
    if (!this.initialized || !this.db) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readwrite");
      const store = transaction.objectStore(this.storeName);
      const request = store.delete(filename);

      request.onsuccess = () => {
        console.log(`[ModelStorage] Deleted ${filename} from IndexedDB.`);
        resolve(true);
      };

      request.onerror = () => {
        console.error(
          `[ModelStorage] Error deleting ${filename}:`,
          request.error
        );
        reject(new Error(`Failed to delete ${filename}: ${request.error}`));
      };
    });
  }

  async clearAll(): Promise<boolean> {
    if (!this.initialized || !this.db) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readwrite");
      const store = transaction.objectStore(this.storeName);
      const request = store.clear();

      request.onsuccess = () => {
        console.log("[ModelStorage] Cleared all models from IndexedDB.");
        resolve(true);
      };

      request.onerror = () => {
        console.error(
          "[ModelStorage] Error clearing all models:",
          request.error
        );
        reject(new Error(`Failed to clear all models: ${request.error}`));
      };
    });
  }

  // IndexedDBのクリーンアップ
  async close(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
      this.initialized = false;
      console.log("[ModelStorage] IndexedDB connection closed.");
    }
  }

  // 使用可能な容量情報を取得（可能な場合）
  async getStorageQuota(): Promise<{ used: number; total: number } | null> {
    if ("storage" in navigator && "estimate" in navigator.storage) {
      try {
        const estimate = await navigator.storage.estimate();
        return {
          used: estimate.usage || 0,
          total: estimate.quota || 0,
        };
      } catch (error) {
        console.warn("[ModelStorage] Failed to get storage quota:", error);
        return null;
      }
    }
    return null;
  }
}
