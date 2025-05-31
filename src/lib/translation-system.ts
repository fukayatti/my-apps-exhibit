import * as ort from "onnxruntime-web";
import { ModelStorage } from "./model-storage";
import { ModelDownloader } from "./model-downloader";
import { SMALL100TokenizerJS } from "./tokenizer";
import { TranslationConfig, StatusMessage, ModelInfo } from "./types";

export class TranslationSystem {
  private session: ort.InferenceSession | null = null;
  private tokenizer: SMALL100TokenizerJS | null = null;
  private config: TranslationConfig | null = null;
  private isLoaded = false;
  private downloader: ModelDownloader;
  private storage: ModelStorage;
  private storageInitialized = false;

  constructor() {
    this.downloader = new ModelDownloader();
    this.storage = new ModelStorage();
    this.initializeOnnxRuntime();
  }

  private initializeOnnxRuntime(): void {
    // ONNX Runtime Webの設定
    ort.env.wasm.numThreads = 1; // スレッド数を1に設定してCORS問題を回避
    ort.env.wasm.simd = true; // SIMD最適化を有効化
    ort.env.webgpu.profiling = { mode: "off" }; // WebGPUプロファイリングを無効化
    ort.env.logLevel = "warning"; // ログレベルを設定

    // WebGPUサポートチェック
    if (typeof navigator !== "undefined" && "gpu" in navigator) {
      console.log("✓ WebGPUサポートが検出されました");
    } else {
      console.log(
        "⚠️ WebGPUサポートが検出されませんでした。WASMにフォールバックします。"
      );
    }
  }

  async initStorage(): Promise<void> {
    if (!this.storageInitialized) {
      await this.storage.init();
      this.storageInitialized = true;
    }
  }

  async downloadAndLoadModel(
    onStatusUpdate: (status: StatusMessage) => void,
    onProgress: (progress: number) => void,
    onFileStatusUpdate: (filename: string, progress: number) => void,
    onFileComplete: (filename: string, size: number) => void
  ): Promise<void> {
    try {
      await this.initStorage();

      onStatusUpdate({ message: "キャッシュを確認中...", type: "loading" });

      // キャッシュされたモデルをチェック
      const requiredFiles = [
        "model.onnx",
        "vocab.json",
        "sentencepiece.bpe.model",
        "config.json",
      ];
      const cachedFiles: Record<string, ArrayBuffer> = {};
      let allCached = true;

      for (const filename of requiredFiles) {
        const hasFile = await this.storage.hasModel(filename);
        if (hasFile) {
          const info = await this.storage.getModelInfo(filename);
          if (info) {
            onFileComplete(filename, info.size);
            const buffer = await this.storage.getModel(filename);
            if (buffer) {
              cachedFiles[filename] = buffer;
              console.log(
                `✓ ${filename} をキャッシュから読み込み (${(
                  info.size /
                  (1024 * 1024)
                ).toFixed(1)}MB)`
              );
            }
          }
        } else {
          allCached = false;
          break;
        }
      }

      let files: Record<string, ArrayBuffer>;
      if (allCached) {
        onStatusUpdate({
          message: "キャッシュからモデルを読み込み中...",
          type: "loading",
        });
        onProgress(100);
        files = cachedFiles;
      } else {
        onStatusUpdate({
          message: "モデルファイルをダウンロード中...",
          type: "loading",
        });

        files = await this.downloader.downloadAllFiles(
          (progress, currentFile, fileProgress) => {
            onProgress(progress);
            onFileStatusUpdate(currentFile, fileProgress);
          },
          async (filename, size) => {
            onFileComplete(filename, size);
            // ダウンロードしたファイルをIndexedDBに保存
            const buffer = this.downloader.getDownloadedFile(filename);
            if (buffer) {
              await this.storage.saveModel(filename, buffer);
              console.log(`💾 ${filename} をIndexedDBに保存しました`);
            }
          }
        );
      }

      onStatusUpdate({ message: "ファイルを処理中...", type: "loading" });

      // 設定ファイルの読み込み
      const configBuffer = files["config.json"];
      const configText = new TextDecoder().decode(configBuffer);
      this.config = JSON.parse(configText);

      // 語彙ファイルの読み込み
      const vocabBuffer = files["vocab.json"];

      // トークナイザーの初期化
      this.tokenizer = new SMALL100TokenizerJS();
      await this.tokenizer.loadVocab(vocabBuffer);

      // SentencePieceモデルの読み込み
      await this.tokenizer.loadSentencePiece();

      // ONNXモデルの読み込み
      const modelBuffer = files["model.onnx"];

      // ONNX Runtime用の設定（WebGPUを優先、フォールバックでWASM）
      const sessionOptions: ort.InferenceSession.SessionOptions = {
        executionProviders: [
          "webgpu", // WebGPUを使用
          "wasm", // WebGPUが利用できない場合のフォールバック
        ],
        graphOptimizationLevel: "all",
        executionMode: "sequential",
        enableMemPattern: false,
        enableCpuMemArena: false,
        logId: "translation-session",
        logSeverityLevel: 2,
      };

      this.session = await ort.InferenceSession.create(
        modelBuffer,
        sessionOptions
      );

      if (this.config && this.tokenizer) {
        this.tokenizer.setDecoderStartTokenId(
          this.config.decoder_start_token_id
        );
        this.tokenizer.setEosTokenId(this.config.eos_token_id);
      }

      this.isLoaded = true;
      onStatusUpdate({
        message: "モデルの読み込みが完了しました！",
        type: "success",
      });
    } catch (error) {
      console.error("モデル読み込みエラー:", error);
      let errorMessage = "モデルの読み込みに失敗しました。";

      if (error instanceof Error) {
        if (error.message.includes("Failed to fetch")) {
          errorMessage =
            "ネットワークエラー: モデルファイルのダウンロードに失敗しました。インターネット接続を確認してください。";
        } else if (
          error.message.includes("out of memory") ||
          error.message.includes("1869662496")
        ) {
          errorMessage =
            "メモリ不足: ブラウザのメモリが不足しています。他のタブを閉じるか、デバイスを再起動してお試しください。";
        } else if (error.message.includes("CORS")) {
          errorMessage =
            "CORS エラー: ブラウザのセキュリティ制限により、ファイルの読み込みに失敗しました。Chrome またはFirefoxをお試しください。";
        } else if (
          error.message.includes("webgpu") ||
          error.message.includes("WebGPU")
        ) {
          errorMessage =
            "WebGPU エラー: WebGPUの初期化に失敗しました。ChromeまたはEdgeの最新版をお試しください。";
        } else {
          errorMessage = `エラー: ${error.message}`;
        }
      }

      onStatusUpdate({ message: errorMessage, type: "error" });
      throw error;
    }
  }

  async translate(inputText: string, maxLength = 128): Promise<string> {
    if (!this.isLoaded || !this.session || !this.tokenizer) {
      throw new Error("モデルが読み込まれていません");
    }

    try {
      // トークン化
      const encodings = this.tokenizer.tokenize(inputText);
      const inputIds = new BigInt64Array(
        encodings.input_ids[0].map((id) => BigInt(id))
      );

      // デコーダーの初期トークン
      const decoderStartToken = this.tokenizer.decoderStartToken;
      if (decoderStartToken === null) {
        throw new Error("Decoder start token not set");
      }

      const generatedIds = [decoderStartToken];

      // 逐次生成
      for (let step = 0; step < maxLength; step++) {
        const decoderInputIds = new BigInt64Array(
          generatedIds.map((id) => BigInt(id))
        );

        // ONNX推論実行
        const feeds: Record<string, ort.Tensor> = {
          input_ids: new ort.Tensor("int64", inputIds, [1, inputIds.length]),
          decoder_input_ids: new ort.Tensor("int64", decoderInputIds, [
            1,
            decoderInputIds.length,
          ]),
        };

        const outputs = await this.session.run(feeds);
        const logitsOutput = outputs.logits;

        if (!logitsOutput || !logitsOutput.data) {
          throw new Error("Invalid model output");
        }

        const logits = logitsOutput.data as Float32Array;

        // 最後のトークンの確率分布から次のトークンを選択
        const vocabSize = logits.length / decoderInputIds.length;
        const lastTokenLogits = logits.slice(-vocabSize);
        const nextTokenId = this.argmax(lastTokenLogits);

        generatedIds.push(nextTokenId);

        // EOSトークンで終了
        const eosToken = this.tokenizer.eosToken;
        if (eosToken !== null && nextTokenId === eosToken) {
          break;
        }
      }

      // デコード
      let decodedIds = generatedIds.slice(1); // BOSを除去
      const eosToken = this.tokenizer.eosToken;
      if (eosToken !== null && decodedIds.includes(eosToken)) {
        const eosIndex = decodedIds.indexOf(eosToken);
        decodedIds = decodedIds.slice(0, eosIndex);
      }

      const translatedText = this.tokenizer.decode(decodedIds, true);
      return translatedText;
    } catch (error) {
      console.error("翻訳エラー:", error);
      throw error;
    }
  }

  private argmax(array: Float32Array): number {
    let maxIndex = 0;
    let maxValue = array[0];
    for (let i = 1; i < array.length; i++) {
      if (array[i] > maxValue) {
        maxValue = array[i];
        maxIndex = i;
      }
    }
    return maxIndex;
  }

  async getCacheInfo(): Promise<ModelInfo[]> {
    await this.initStorage();
    return await this.storage.getAllModelInfo();
  }

  async clearCache(): Promise<void> {
    await this.initStorage();
    await this.storage.clearAll();
    this.isLoaded = false;
  }

  get isModelLoaded(): boolean {
    return this.isLoaded;
  }

  getRequiredFiles(): string[] {
    return this.downloader.getFiles().map((file) => file.name);
  }
}
