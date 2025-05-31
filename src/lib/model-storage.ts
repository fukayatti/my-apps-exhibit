import { ModelInfo, FileStatus } from "./types";

export class ModelStorage {
  private dbName = "TranslationModelDB";
  private dbVersion = 1;
  private storeName = "models";
  private db: IDBDatabase | null = null;

  async init(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          const store = db.createObjectStore(this.storeName, {
            keyPath: "filename",
          });
          store.createIndex("filename", "filename", { unique: true });
        }
      };
    });
  }

  async saveModel(
    filename: string,
    buffer: ArrayBuffer,
    metadata: Record<string, unknown> = {}
  ): Promise<IDBValidKey> {
    if (!this.db) throw new Error("Database not initialized");

    const transaction = this.db.transaction([this.storeName], "readwrite");
    const store = transaction.objectStore(this.storeName);

    const modelData = {
      filename: filename,
      data: buffer,
      timestamp: Date.now(),
      size: buffer.byteLength,
      ...metadata,
    };

    return new Promise((resolve, reject) => {
      const request = store.put(modelData);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async getModel(filename: string): Promise<ArrayBuffer | null> {
    if (!this.db) throw new Error("Database not initialized");

    const transaction = this.db.transaction([this.storeName], "readonly");
    const store = transaction.objectStore(this.storeName);

    return new Promise((resolve, reject) => {
      const request = store.get(filename);
      request.onsuccess = () => {
        const result = request.result;
        resolve(result ? result.data : null);
      };
      request.onerror = () => reject(request.error);
    });
  }

  async hasModel(filename: string): Promise<boolean> {
    if (!this.db) throw new Error("Database not initialized");

    const transaction = this.db.transaction([this.storeName], "readonly");
    const store = transaction.objectStore(this.storeName);

    return new Promise((resolve, reject) => {
      const request = store.get(filename);
      request.onsuccess = () => resolve(!!request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async getModelInfo(filename: string): Promise<FileStatus | null> {
    if (!this.db) throw new Error("Database not initialized");

    const transaction = this.db.transaction([this.storeName], "readonly");
    const store = transaction.objectStore(this.storeName);

    return new Promise((resolve, reject) => {
      const request = store.get(filename);
      request.onsuccess = () => {
        const result = request.result;
        if (result) {
          resolve({
            filename: result.filename,
            size: result.size,
            timestamp: result.timestamp,
          });
        } else {
          resolve(null);
        }
      };
      request.onerror = () => reject(request.error);
    });
  }

  async getAllModelInfo(): Promise<FileStatus[]> {
    if (!this.db) throw new Error("Database not initialized");

    const transaction = this.db.transaction([this.storeName], "readonly");
    const store = transaction.objectStore(this.storeName);

    return new Promise((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => {
        const results = request.result.map(
          (item: { filename: string; size: number; timestamp: number }) => ({
            filename: item.filename,
            size: item.size,
            timestamp: item.timestamp,
          })
        );
        resolve(results);
      };
      request.onerror = () => reject(request.error);
    });
  }

  async getAllFilesInfo(): Promise<FileStatus[]> {
    if (!this.db) throw new Error("Database not initialized");

    const transaction = this.db.transaction([this.storeName], "readonly");
    const store = transaction.objectStore(this.storeName);

    return new Promise((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => {
        const results = request.result.map(
          (item: { filename: string; size: number; timestamp: number }) => ({
            filename: item.filename,
            size: item.size,
            timestamp: item.timestamp,
          })
        );
        resolve(results);
      };
      request.onerror = () => reject(request.error);
    });
  }

  async deleteModel(filename: string): Promise<boolean> {
    if (!this.db) throw new Error("Database not initialized");

    const transaction = this.db.transaction([this.storeName], "readwrite");
    const store = transaction.objectStore(this.storeName);

    return new Promise((resolve, reject) => {
      const request = store.delete(filename);
      request.onsuccess = () => resolve(true);
      request.onerror = () => reject(request.error);
    });
  }

  async clearAll(): Promise<boolean> {
    if (!this.db) throw new Error("Database not initialized");

    const transaction = this.db.transaction([this.storeName], "readwrite");
    const store = transaction.objectStore(this.storeName);

    return new Promise((resolve, reject) => {
      const request = store.clear();
      request.onsuccess = () => resolve(true);
      request.onerror = () => reject(request.error);
    });
  }
}
