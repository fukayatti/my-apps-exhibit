import * as ort from "onnxruntime-web";
import { ModelStorage } from "./model-storage";
import { ModelDownloader } from "./model-downloader";
import { HuggingFaceTokenizer } from "./tokenizer";
import {
  TranslationConfig,
  StatusMessage,
  ModelInfo,
  TokenizerConfig,
  AddedToken,
  FileStatus,
} from "./types";

export class TranslationSystem {
  private session: ort.InferenceSession | null = null;
  private tokenizer: HuggingFaceTokenizer | null = null;
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
    ort.env.logLevel = "error"; // エラーレベルのみ表示

    // WebNNサポートチェック
    if (typeof navigator !== "undefined" && "ml" in navigator) {
      console.log("✓ WebNNサポートが検出されました");
    } else {
      console.log(
        "⚠️ WebNNサポートが検出されませんでした。WebGPUにフォールバックします。"
      );
    }
    // WebGPUサポートチェック
    if (typeof navigator !== "undefined" && "gpu" in navigator) {
      console.log("✓ WebGPUサポートが検出されました");
    }
  }

  async initStorage(): Promise<void> {
    if (!this.storageInitialized) {
      await this.storage.init();
      this.storageInitialized = true;
    }
  }

  async downloadModel(
    onProgress: (message: StatusMessage) => void
  ): Promise<void> {
    await this.initStorage();
    onProgress({
      type: "info",
      message: "モデルファイルのダウンロードを開始します...",
    });

    const files = await this.downloader.downloadAllFiles(
      (progress: number, currentFile: string, fileProgress: number) => {
        onProgress({
          type: "progress",
          message: `ファイルダウンロード中: ${currentFile} (${Math.round(
            fileProgress
          )}%)`,
          progress: progress,
        });
      },
      (filename: string, size: number) => {
        onProgress({
          type: "info",
          message: `${filename} のダウンロード完了 (${Math.round(
            size / 1024 / 1024
          )}MB)`,
        });
      }
    );

    onProgress({ type: "info", message: "モデルファイルを保存中..." });
    for (const [name, buffer] of Object.entries(files)) {
      await this.storage.saveModel(name, buffer);
    }
    onProgress({
      type: "success",
      message: "モデルのダウンロードが完了しました。",
    });
  }

  async loadModel(onProgress: (message: StatusMessage) => void): Promise<void> {
    await this.initStorage();
    onProgress({ type: "info", message: "モデルの読み込みを開始します..." });

    try {
      // キャッシュされたモデルをチェック
      const requiredFiles = [
        "model.onnx",
        "vocab.json",
        "tokenizer_config.json",
        "config.json",
      ];
      const files: Record<string, ArrayBuffer> = {};
      let allFilesExist = true;

      for (const fileName of requiredFiles) {
        const fileBuffer = await this.storage.getModel(fileName);
        if (fileBuffer) {
          files[fileName] = fileBuffer;
        } else {
          allFilesExist = false;
          break;
        }
      }

      if (!allFilesExist) {
        onProgress({
          type: "info",
          message:
            "必要なファイルがキャッシュにありません。ダウンロードを開始します...",
        });
        await this.downloadModel(onProgress);
        // ダウンロード後に再度ファイルを読み込む
        for (const fileName of requiredFiles) {
          const fileBuffer = await this.storage.getModel(fileName);
          if (fileBuffer) {
            files[fileName] = fileBuffer;
          } else {
            throw new Error(
              `ファイル ${fileName} がダウンロード後も見つかりません。`
            );
          }
        }
      }

      onProgress({ type: "info", message: "ONNXモデルを初期化中..." });
      const modelBuffer = files["model.onnx"];
      this.session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ["webgpu", "wasm"], // WebGPUを優先、フォールバックでWASM
        graphOptimizationLevel: "all",
      });
      console.log("✓ ONNXセッションの作成完了");
      onProgress({ type: "info", message: "ONNXモデルの初期化完了。" });

      onProgress({ type: "info", message: "トークナイザーを初期化中..." });
      // トークナイザーの初期化
      this.tokenizer = new HuggingFaceTokenizer();

      // vocab.jsonとtokenizer_config.jsonからtokenizer.json形式を生成
      const vocabBuffer = files["vocab.json"];
      const tokenizerConfigBuffer = files["tokenizer_config.json"];

      const tokenizerJsonBuffer = this.createTokenizerJson(
        vocabBuffer,
        tokenizerConfigBuffer
      );
      await this.tokenizer.loadFromBuffer(tokenizerJsonBuffer);

      console.log("✓ トークナイザーの初期化完了");
      onProgress({ type: "info", message: "トークナイザーの初期化完了。" });

      onProgress({ type: "info", message: "設定ファイルを読み込み中..." });
      const configBuffer = files["config.json"];
      this.config = JSON.parse(
        new TextDecoder().decode(configBuffer)
      ) as TranslationConfig;
      console.log("✓ 設定ファイルの読み込み完了");
      onProgress({ type: "info", message: "設定ファイルの読み込み完了。" });

      this.isLoaded = true;
      onProgress({
        type: "success",
        message: "モデルの読み込みが完了しました。",
      });
    } catch (error) {
      console.error("モデル読み込みエラー:", error);
      onProgress({
        type: "error",
        message: `モデルの読み込みに失敗しました: ${error}`,
      });
      throw error;
    }
  }

  async downloadAndLoadModel(
    onStatusUpdate: (status: StatusMessage) => void,
    onProgressUpdate: (progress: number) => void,
    onFileProgress: (filename: string, progress: number) => void,
    onFileComplete: (filename: string, size: number) => void
  ): Promise<void> {
    await this.initStorage();

    // まずキャッシュをチェック
    const requiredFiles = [
      "model.onnx",
      "vocab.json",
      "tokenizer_config.json",
      "config.json",
    ];

    let allFilesExist = true;
    for (const fileName of requiredFiles) {
      const fileBuffer = await this.storage.getModel(fileName);
      if (!fileBuffer) {
        allFilesExist = false;
        break;
      }
    }

    if (!allFilesExist) {
      // ダウンロードが必要
      onStatusUpdate({
        type: "info",
        message: "モデルファイルのダウンロードを開始します...",
      });

      const files = await this.downloader.downloadAllFiles(
        (progress: number, currentFile: string, fileProgress: number) => {
          onProgressUpdate(progress);
          onFileProgress(currentFile, fileProgress);
          onStatusUpdate({
            type: "progress",
            message: `ファイルダウンロード中: ${currentFile} (${Math.round(
              fileProgress
            )}%)`,
            progress: progress,
          });
        },
        (filename: string, size: number) => {
          onFileComplete(filename, size);
          onStatusUpdate({
            type: "info",
            message: `${filename} のダウンロード完了 (${Math.round(
              size / 1024 / 1024
            )}MB)`,
          });
        }
      );

      onStatusUpdate({ type: "info", message: "モデルファイルを保存中..." });
      for (const [name, buffer] of Object.entries(files)) {
        await this.storage.saveModel(name, buffer);
      }
    }

    // モデルを読み込み
    await this.loadModel(onStatusUpdate);
  }

  async translate(
    text: string,
    sourceLang: string,
    targetLang: string,
    onProgress?: (message: StatusMessage) => void
  ): Promise<string> {
    if (!this.isLoaded || !this.session || !this.tokenizer || !this.config) {
      throw new Error(
        "モデルが読み込まれていません。loadModel()を呼び出してください。"
      );
    }

    if (onProgress)
      onProgress({ type: "info", message: "翻訳処理を開始します..." });

    try {
      // ソース言語のトークンIDを設定
      // const srcLangToken = this.tokenizer.getLangToken(sourceLang); // 未使用
      const srcLangId = this.tokenizer.getLangId(sourceLang);
      if (srcLangId === undefined) {
        throw new Error(
          `ソース言語 ${sourceLang} のトークンIDが見つかりません。`
        );
      }
      this.tokenizer.setEosTokenId(srcLangId); // NLLBモデルではeos_token_idがソース言語ID

      if (onProgress)
        onProgress({ type: "info", message: "テキストをトークン化中..." });
      const { input_ids } = this.tokenizer.tokenize(text);
      const inputIdsTensor = new ort.Tensor(
        "int64",
        BigInt64Array.from(input_ids[0].map(BigInt)),
        [1, input_ids[0].length]
      );

      // デコーダーの開始トークンIDを設定
      // const tgtLangToken = this.tokenizer.getLangToken(targetLang); // 未使用
      const tgtLangId = this.tokenizer.getLangId(targetLang);
      if (tgtLangId === undefined) {
        throw new Error(
          `ターゲット言語 ${targetLang} のトークンIDが見つかりません。`
        );
      }
      this.tokenizer.setDecoderStartTokenId(tgtLangId);

      const decoderStartTokenId = this.tokenizer.decoderStartToken;
      if (decoderStartTokenId === null) {
        throw new Error("デコーダー開始トークンIDが取得できません。");
      }

      const feeds: Record<string, ort.Tensor> = {
        input_ids: inputIdsTensor,
        decoder_input_ids: new ort.Tensor(
          "int64",
          BigInt64Array.from([BigInt(decoderStartTokenId)]),
          [1, 1]
        ),
      };

      if (onProgress)
        onProgress({ type: "info", message: "ONNXモデルで推論中..." });

      // Beam Searchを手動で実装 (簡易版)
      const numBeams = this.config.num_beams || 4;
      const maxLength = this.config.max_length || 200;
      const beams: Array<{ tokens: number[]; score: number }> = [
        { tokens: [decoderStartTokenId], score: 0.0 },
      ];
      const completedSequences: Array<{ tokens: number[]; score: number }> = [];

      for (let step = 0; step < maxLength; step++) {
        if (beams.length === 0) break;

        const nextBeams: Array<{ tokens: number[]; score: number }> = [];
        for (const beam of beams) {
          feeds.decoder_input_ids = new ort.Tensor(
            "int64",
            BigInt64Array.from(beam.tokens.map(BigInt)),
            [1, beam.tokens.length]
          );

          const output = await this.session.run(feeds);
          const logits = output.logits.data as Float32Array; // 型アサーション

          // 次のトークンの確率を取得 (最後のトークンのみ)
          const nextTokenLogits = logits.slice(
            (beam.tokens.length - 1) * this.config.vocab_size,
            beam.tokens.length * this.config.vocab_size
          );

          // ソフトマックス関数で確率に変換
          const probabilities = this.softmax(nextTokenLogits);

          // 上位k個のトークンを選択 (k=numBeams)
          const topK = this.getTopK(probabilities, numBeams);

          for (const { index: tokenId, probability } of topK) {
            const newTokens = [...beam.tokens, tokenId];
            const newScore = beam.score + Math.log(probability); // 対数確率を使用

            if (tokenId === this.tokenizer.eosToken) {
              completedSequences.push({ tokens: newTokens, score: newScore });
            } else {
              nextBeams.push({ tokens: newTokens, score: newScore });
            }
          }
        }

        // ビームをスコアでソートし、上位numBeams個を保持
        beams.length = 0; // beamsをクリア
        beams.push(
          ...nextBeams.sort((a, b) => b.score - a.score).slice(0, numBeams)
        );

        if (onProgress && step % 10 === 0) {
          onProgress({
            type: "info",
            message: `推論中 (ステップ ${step + 1}/${maxLength})...`,
          });
        }
      }

      // 最もスコアの高い完了シーケンスを選択
      const bestSequence =
        completedSequences.sort((a, b) => b.score - a.score)[0]?.tokens ||
        beams.sort((a, b) => b.score - a.score)[0]?.tokens;

      if (!bestSequence) {
        throw new Error("翻訳結果が生成されませんでした。");
      }

      if (onProgress)
        onProgress({ type: "info", message: "トークンをデコード中..." });
      const translatedText = this.tokenizer.decode(bestSequence, true);

      if (onProgress)
        onProgress({ type: "success", message: "翻訳処理が完了しました。" });
      return translatedText;
    } catch (error) {
      console.error("翻訳エラー:", error);
      if (onProgress)
        onProgress({
          type: "error",
          message: `翻訳に失敗しました: ${error}`,
        });
      throw error;
    }
  }

  getModelInfo(): ModelInfo | null {
    if (!this.isLoaded || !this.config) return null;
    return {
      model_name: this.config._name_or_path || "N/A",
      vocab_size: this.config.vocab_size || 0,
      num_beams: this.config.num_beams || 0,
      max_length: this.config.max_length || 0,
      architectures: this.config.architectures || [],
    };
  }

  isModelLoaded(): boolean {
    return this.isLoaded;
  }

  async clearCache(): Promise<void> {
    await this.initStorage();
    await this.storage.clearAll();
    this.isLoaded = false;
    this.session = null;
    this.tokenizer = null;
    this.config = null;
  }

  async getCacheInfo(): Promise<FileStatus[]> {
    await this.initStorage();
    return this.storage.getAllFilesInfo();
  }

  private softmax(array: Float32Array): Float32Array {
    const maxLogit = Math.max(...Array.from(array));
    const result = new Float32Array(array.length);
    let sumExps = 0;

    // 最初にexpを計算してsumを求める
    for (let i = 0; i < array.length; i++) {
      result[i] = Math.exp(array[i] - maxLogit);
      sumExps += result[i];
    }

    // 正規化
    for (let i = 0; i < result.length; i++) {
      result[i] = result[i] / sumExps;
    }

    return result;
  }

  private getTopK(
    probabilities: Float32Array,
    k: number
  ): Array<{ index: number; probability: number }> {
    return Array.from(probabilities)
      .map((probability, index) => ({ index, probability }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, k);
  }

  // private argmax(array: Float32Array): number {
  //   let maxIndex = 0;
  //   let maxValue = array[0];
  //   for (let i = 1; i < array.length; i++) {
  //     if (array[i] > maxValue) {
  //       maxValue = array[i];
  //       maxIndex = i;
  //     }
  //   }
  //   return maxIndex;
  // }

  private createTokenizerJson(
    vocabBuffer: ArrayBuffer,
    tokenizerConfigBuffer: ArrayBuffer
  ): ArrayBuffer {
    try {
      // vocab.jsonとtokenizer_config.jsonを解析
      const vocab = JSON.parse(new TextDecoder().decode(vocabBuffer));
      const tokenizerConfig: TokenizerConfig = JSON.parse(
        new TextDecoder().decode(tokenizerConfigBuffer)
      );

      // tokenizer.json形式を生成
      const tokenizerJson = {
        version: "1.0",
        truncation: null,
        padding: null,
        added_tokens: [] as AddedToken[], // 型を指定
        normalizer: {
          // NLLB-200の一般的な設定
          type: "Sequence",
          normalizers: [
            { type: "Prepend", prepend: " " },
            { type: "Replace", pattern: { String: " " }, content: " " },
          ],
        },
        pre_tokenizer: {
          // NLLB-200の一般的な設定
          type: "Metaspace",
          replacement: " ",
          add_prefix_space: true,
          prepend_scheme: "always",
        },
        model: {
          type: "BPE", // SentencePieceはBPEの一種として扱われることが多い
          dropout: null,
          unk_token: "<unk>", // tokenizerConfigから取得するべき
          continuing_subword_prefix: null, // SentencePieceでは通常使用しない
          end_of_word_suffix: null, // SentencePieceでは通常使用しない
          fuse_unk: false,
          byte_fallback: false, // SentencePieceでは通常使用しない
          vocab: vocab,
          merges: [], // SentencePieceの場合、マージルールは.modelファイル内。ここでは空
        },
        decoder: {
          // NLLB-200の一般的な設定
          type: "Metaspace",
          replacement: " ",
          add_prefix_space: true,
          prepend_scheme: "always",
        },
        post_processor: null, // NLLB-200では通常シンプルなポストプロセス
      };

      // tokenizerConfigから情報をマージ
      if (tokenizerConfig.added_tokens) {
        tokenizerJson.added_tokens = tokenizerConfig.added_tokens;
      }

      if (tokenizerConfig.special_tokens_map) {
        const specialTokensMap = tokenizerConfig.special_tokens_map;
        const addedTokensMap = new Map<string, AddedToken>(
          tokenizerJson.added_tokens.map((t) => [t.content, t])
        );

        for (const [, tokenValueObj] of Object.entries(specialTokensMap)) {
          let tokenContent: string;
          let tokenId: number | undefined = undefined;

          if (typeof tokenValueObj === "string") {
            tokenContent = tokenValueObj;
          } else if (
            typeof tokenValueObj === "object" &&
            tokenValueObj.content
          ) {
            tokenContent = tokenValueObj.content;
            tokenId = tokenValueObj.id; // idが提供されていれば使用
          } else {
            continue;
          }

          if (
            vocab[tokenContent] !== undefined &&
            !addedTokensMap.has(tokenContent)
          ) {
            addedTokensMap.set(tokenContent, {
              id: tokenId ?? vocab[tokenContent], // idがなければvocabから
              content: tokenContent,
              single_word: false, // デフォルト値
              lstrip: false, // デフォルト値
              rstrip: false, // デフォルト値
              normalized: false, // デフォルト値
              special: true,
            });
          } else if (
            addedTokensMap.has(tokenContent) &&
            tokenId !== undefined
          ) {
            // 既存のトークンIDを更新 (もしあれば)
            const existingToken = addedTokensMap.get(tokenContent);
            if (existingToken && tokenId !== undefined)
              existingToken.id = tokenId;
          }
        }
        tokenizerJson.added_tokens = Array.from(addedTokensMap.values());
      }

      if (tokenizerConfig.unk_token) {
        if (typeof tokenizerConfig.unk_token === "string") {
          tokenizerJson.model.unk_token = tokenizerConfig.unk_token;
        } else if (
          typeof tokenizerConfig.unk_token === "object" &&
          tokenizerConfig.unk_token.content
        ) {
          tokenizerJson.model.unk_token = tokenizerConfig.unk_token.content;
        }
      }

      // 必須の特殊トークンがadded_tokensに含まれているか確認し、なければ追加
      const ensureSpecialToken = (content: string, defaultId: number) => {
        if (!tokenizerJson.added_tokens.some((t) => t.content === content)) {
          tokenizerJson.added_tokens.push({
            id: vocab[content] ?? defaultId,
            content: content,
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
            special: true,
          } as AddedToken);
        }
      };

      ensureSpecialToken("<pad>", 0);
      ensureSpecialToken("</s>", 1);
      ensureSpecialToken("<s>", 2);
      ensureSpecialToken("<unk>", vocab[tokenizerJson.model.unk_token] ?? 3);

      const tokenizerJsonString = JSON.stringify(tokenizerJson, null, 2);
      const uint8Array = new TextEncoder().encode(tokenizerJsonString);
      return uint8Array.buffer.slice(
        uint8Array.byteOffset,
        uint8Array.byteOffset + uint8Array.byteLength
      );
    } catch (error) {
      console.error("tokenizer.json生成エラー:", error);
      throw new Error(`tokenizer.json生成に失敗: ${error}`);
    }
  }
}
