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
    onProgress: (message: StatusMessage) => void,
    onFileProgress: (
      filename: string,
      progress: number,
      loaded: number,
      total: number
    ) => void,
    onFileComplete: (filename: string, size: number) => void
  ): Promise<void> {
    await this.initStorage();
    const statusStart: StatusMessage = {
      type: "info",
      message: "モデルファイルのダウンロードを開始します...",
    };
    console.log(`[TranslationSystem] ${statusStart.message}`);
    onProgress(statusStart);

    try {
      const files = await this.downloader.downloadAllFiles(
        (
          progress: number,
          currentFile: string,
          fileProg: number,
          loadedBytes?: number,
          totalBytes?: number
        ) => {
          const progressMsg: StatusMessage = {
            type: "progress",
            message: `ファイルダウンロード中: ${currentFile} (${Math.round(
              fileProg
            )}%)`,
            progress: progress,
          };
          // console.log(`[TranslationSystem] ${progressMsg.message} - Overall: ${progress.toFixed(2)}%`);
          onProgress(progressMsg);
          if (loadedBytes !== undefined && totalBytes !== undefined) {
            onFileProgress(currentFile, fileProg, loadedBytes, totalBytes);
          } else {
            onFileProgress(currentFile, fileProg, 0, 0); // totalが不明な場合もあるため
          }
        },
        (filename: string, size: number) => {
          const completeMsg: StatusMessage = {
            type: "info",
            message: `${filename} のダウンロード完了 (${(
              size /
              1024 /
              1024
            ).toFixed(2)}MB)`,
          };
          console.log(`[TranslationSystem] ${completeMsg.message}`);
          onProgress(completeMsg);
          onFileComplete(filename, size);
        }
      );

      const statusSaving: StatusMessage = {
        type: "info",
        message: "ダウンロードしたモデルファイルを保存中...",
      };
      console.log(`[TranslationSystem] ${statusSaving.message}`);
      onProgress(statusSaving);

      for (const [name, buffer] of Object.entries(files)) {
        console.log(
          `[TranslationSystem] 保存中: ${name} (${(
            buffer.byteLength /
            1024 /
            1024
          ).toFixed(2)}MB)`
        );
        await this.storage.saveModel(name, buffer);
        console.log(`[TranslationSystem] 保存完了: ${name}`);
      }
      const statusSuccess: StatusMessage = {
        type: "success",
        message: "モデルのダウンロードと保存が完了しました。",
      };
      console.log(`[TranslationSystem] ${statusSuccess.message}`);
      onProgress(statusSuccess);
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      console.error(
        "[TranslationSystem] downloadModelでエラー発生:",
        errorMessage,
        error
      );
      const statusError: StatusMessage = {
        type: "error",
        message: `モデルのダウンロードに失敗しました: ${errorMessage}`,
      };
      onProgress(statusError);
      throw error; // エラーを再スローして呼び出し元で処理できるようにする
    }
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
        // downloadModelのシグネチャ変更に合わせて呼び出しを修正
        // このコンテキストでは個別のファイル進捗は不要なため、ダミーのコールバックを渡す
        await this.downloadModel(
          onProgress,
          () => {},
          () => {}
        );
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
        message: `モデルの読み込みに失敗しました: ${
          error instanceof Error ? error.message : String(error)
        }`,
      });
      console.error("[TranslationSystem] loadModelでエラー発生:", error);
      throw error;
    }
  }

  async downloadAndLoadModel(
    onStatusUpdate: (status: StatusMessage) => void,
    onOverallProgressUpdate: (progress: number) => void, // 全体進捗
    onFileProgressUpdate: (
      filename: string,
      progress: number,
      loadedBytes: number,
      totalBytes: number
    ) => void, // 個別ファイル進捗
    onFileCompleteUpdate: (filename: string, size: number) => void // 個別ファイル完了
  ): Promise<void> {
    await this.initStorage();
    console.log("[TranslationSystem] downloadAndLoadModel開始");

    const requiredFiles = [
      "model.onnx",
      "vocab.json",
      "tokenizer_config.json",
      "config.json",
    ];

    let allFilesExist = true;
    const filesToDownload: string[] = [];
    for (const fileName of requiredFiles) {
      const fileBuffer = await this.storage.getModel(fileName);
      if (!fileBuffer) {
        allFilesExist = false;
        filesToDownload.push(fileName);
        console.log(
          `[TranslationSystem] キャッシュにファイルなし: ${fileName}`
        );
      } else {
        console.log(
          `[TranslationSystem] キャッシュにファイルあり: ${fileName}`
        );
        // キャッシュにあるファイルに対しても完了イベントを発火させる
        const info = await this.storage.getModelInfo(fileName); // getFileInfo を getModelInfo に変更
        if (info) {
          onFileCompleteUpdate(fileName, info.size);
          // 個別ファイルの進捗も100%として通知
          onFileProgressUpdate(fileName, 100, info.size, info.size);
        }
      }
    }
    // 全体進捗を更新 (キャッシュヒット分)
    const cachedFilesCount = requiredFiles.length - filesToDownload.length;
    if (requiredFiles.length > 0) {
      onOverallProgressUpdate((cachedFilesCount / requiredFiles.length) * 100);
    }

    if (!allFilesExist) {
      const downloadStatus: StatusMessage = {
        type: "info",
        message: `必要なモデルファイル (${filesToDownload.join(
          ", "
        )}) のダウンロードを開始します...`,
      };
      console.log(`[TranslationSystem] ${downloadStatus.message}`);
      onStatusUpdate(downloadStatus);

      try {
        // ModelDownloaderのfilesを実際にダウンロードが必要なファイルに絞るか、
        // downloadAllFilesが特定のファイルのみダウンロードする機能を持つように変更する必要がある。
        // ここでは、downloaderが設定された全ファイルをダウンロードしようとする前提で進めるが、
        // 理想的にはdownloader.downloadSpecificFiles(filesToDownload, ...) のような形が良い。
        // 現状のdownloader.downloadAllFilesは固定リストをダウンロードするため、
        // キャッシュチェックとダウンロードのロジックが少し冗長になる。
        // 今回はdownloaderのインターフェースは変更せず、TranslationSystem側で対応。

        const files = await this.downloader.downloadAllFiles(
          (
            overallProgress: number,
            currentFile: string,
            fileSpecificProgress: number,
            loadedBytes?: number,
            totalBytes?: number
          ) => {
            // downloaderからの進捗は全ファイルに対するものなので、
            // UI側の全体進捗とは別に計算し直す必要があるかもしれない。
            // ここではdownloaderからのoverallProgressをそのまま使う。
            onOverallProgressUpdate(overallProgress);
            if (loadedBytes !== undefined && totalBytes !== undefined) {
              onFileProgressUpdate(
                currentFile,
                fileSpecificProgress,
                loadedBytes,
                totalBytes
              );
            } else {
              onFileProgressUpdate(currentFile, fileSpecificProgress, 0, 0);
            }
            const progressMsg: StatusMessage = {
              type: "progress",
              message: `ファイルダウンロード中: ${currentFile} (${Math.round(
                fileSpecificProgress
              )}%)`,
              progress: overallProgress,
            };
            onStatusUpdate(progressMsg);
          },
          (filename: string, size: number) => {
            onFileCompleteUpdate(filename, size);
            const completeMsg: StatusMessage = {
              type: "info",
              message: `${filename} のダウンロード完了 (${(
                size /
                1024 /
                1024
              ).toFixed(2)}MB)`,
            };
            console.log(`[TranslationSystem] ${completeMsg.message}`);
            onStatusUpdate(completeMsg);
          }
        );

        const savingStatus: StatusMessage = {
          type: "info",
          message: "ダウンロードしたモデルファイルを保存中...",
        };
        console.log(`[TranslationSystem] ${savingStatus.message}`);
        onStatusUpdate(savingStatus);
        for (const [name, buffer] of Object.entries(files)) {
          // filesToDownloadに含まれるファイルのみ保存する（downloaderが全ファイル返す場合）
          // もしdownloaderが指定ファイルのみ返すならこのチェックは不要
          if (filesToDownload.includes(name)) {
            console.log(
              `[TranslationSystem] 保存中: ${name} (${(
                buffer.byteLength /
                1024 /
                1024
              ).toFixed(2)}MB)`
            );
            await this.storage.saveModel(name, buffer);
            console.log(`[TranslationSystem] 保存完了: ${name}`);
          }
        }
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        console.error(
          "[TranslationSystem] downloadAndLoadModel - ダウンロード中にエラー発生:",
          errorMessage,
          error
        );
        const statusError: StatusMessage = {
          type: "error",
          message: `モデルのダウンロードに失敗しました: ${errorMessage}`,
        };
        onStatusUpdate(statusError);
        throw error; // エラーを再スロー
      }
    } else {
      console.log(
        "[TranslationSystem] 全てのモデルファイルがキャッシュに存在します。"
      );
      onOverallProgressUpdate(100); // 全てキャッシュヒットなら全体進捗100%
      const cacheMsg: StatusMessage = {
        type: "info",
        message: "全てのモデルファイルがキャッシュに存在します。",
      };
      onStatusUpdate(cacheMsg);
    }

    // モデルを読み込み
    console.log("[TranslationSystem] モデルの読み込みを開始します...");
    await this.loadModel(onStatusUpdate);
    console.log("[TranslationSystem] downloadAndLoadModel完了");
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

    if (onProgress) {
      const startMsg: StatusMessage = {
        type: "info",
        message: "翻訳処理を開始します...",
      };
      console.log(`[TranslationSystem][translate] ${startMsg.message}`);
      onProgress(startMsg);
    }

    try {
      console.log(
        `[TranslationSystem][translate] 入力テキスト: "${text}", ソース言語: ${sourceLang}, ターゲット言語: ${targetLang}`
      );
      // ソース言語のトークンIDを設定
      const srcLangId = this.tokenizer.getLangId(sourceLang);
      if (srcLangId === undefined) {
        throw new Error(
          `ソース言語 ${sourceLang} のトークンIDが見つかりません。利用可能な言語: ${Object.keys(
            this.config?.lang_to_id || {}
          ).join(", ")}`
        );
      }
      this.tokenizer.setEosTokenId(srcLangId);
      console.log(
        `[TranslationSystem][translate] ソース言語ID (${sourceLang}): ${srcLangId}, EOSトークンIDを ${srcLangId} に設定`
      );

      if (onProgress) {
        const tokenizeMsg: StatusMessage = {
          type: "info",
          message: "テキストをトークン化中...",
        };
        console.log(`[TranslationSystem][translate] ${tokenizeMsg.message}`);
        onProgress(tokenizeMsg);
      }
      const { input_ids } = this.tokenizer.tokenize(text);
      const inputIdsTensor = new ort.Tensor(
        "int64",
        BigInt64Array.from(input_ids[0].map(BigInt)),
        [1, input_ids[0].length]
      );
      console.log(
        `[TranslationSystem][translate] トークン化結果 (input_ids): ${
          input_ids[0]
        }, Tensor shape: [${inputIdsTensor.dims.join(",")}]`
      );

      // デコーダーの開始トークンIDを設定
      const tgtLangId = this.tokenizer.getLangId(targetLang);
      if (tgtLangId === undefined) {
        throw new Error(
          `ターゲット言語 ${targetLang} のトークンIDが見つかりません。利用可能な言語: ${Object.keys(
            this.config?.lang_to_id || {}
          ).join(", ")}`
        );
      }
      this.tokenizer.setDecoderStartTokenId(tgtLangId);
      console.log(
        `[TranslationSystem][translate] ターゲット言語ID (${targetLang}): ${tgtLangId}, Decoder StartトークンIDを ${tgtLangId} に設定`
      );

      const decoderStartTokenId = this.tokenizer.decoderStartToken;
      if (decoderStartTokenId === null) {
        throw new Error("デコーダー開始トークンIDが取得できません。");
      }
      console.log(
        `[TranslationSystem][translate] 実際のDecoder StartトークンID: ${decoderStartTokenId}`
      );

      const feeds: Record<string, ort.Tensor> = {
        input_ids: inputIdsTensor,
        decoder_input_ids: new ort.Tensor(
          "int64",
          BigInt64Array.from([BigInt(decoderStartTokenId)]),
          [1, 1]
        ),
      };

      if (onProgress) {
        const inferMsg: StatusMessage = {
          type: "info",
          message: "ONNXモデルで推論中...",
        };
        console.log(`[TranslationSystem][translate] ${inferMsg.message}`);
        onProgress(inferMsg);
      }

      // Beam Searchを手動で実装 (簡易版)
      const numBeams = this.config.num_beams || 4;
      const maxLength = this.config.max_length || 200;
      let beams: Array<{
        tokens: number[];
        score: number;
        completed: boolean;
      }> = [{ tokens: [decoderStartTokenId], score: 0.0, completed: false }];
      const completedSequences: Array<{
        tokens: number[];
        score: number;
        completed: boolean;
      }> = [];
      console.log(
        `[TranslationSystem][translate] Beam Search開始: numBeams=${numBeams}, maxLength=${maxLength}`
      );

      for (let step = 0; step < maxLength; step++) {
        if (beams.every((b) => b.completed)) {
          console.log(
            `[TranslationSystem][translate] Beam Search: 全てのビームが完了 (ステップ ${step})`
          );
          break;
        }
        if (beams.length === 0 && completedSequences.length === 0 && step > 0) {
          console.warn(
            `[TranslationSystem][translate] Beam Search: 有効なビームがありません (ステップ ${step})`
          );
          break;
        }

        const nextBeamsAccumulator: Array<{
          tokens: number[];
          score: number;
          completed: boolean;
        }> = [];
        for (const beam of beams) {
          if (beam.completed) {
            nextBeamsAccumulator.push(beam); // 完了したビームはそのまま次へ
            continue;
          }

          feeds.decoder_input_ids = new ort.Tensor(
            "int64",
            BigInt64Array.from(beam.tokens.map(BigInt)),
            [1, beam.tokens.length]
          );

          const output = await this.session.run(feeds);
          const logits = output.logits.data as Float32Array;

          const nextTokenLogits = logits.slice(
            (beam.tokens.length - 1) * this.config.vocab_size,
            beam.tokens.length * this.config.vocab_size
          );

          const probabilities = this.softmax(nextTokenLogits);
          const topK = this.getTopK(probabilities, numBeams);

          for (const { index: tokenId, probability } of topK) {
            if (probability === 0) continue; // 確率0のトークンは無視

            const newTokens = [...beam.tokens, tokenId];
            const newScore = beam.score + Math.log(probability); // 対数確率

            if (
              tokenId === this.tokenizer.eosToken ||
              newTokens.length >= maxLength
            ) {
              completedSequences.push({
                tokens: newTokens,
                score: newScore,
                completed: true,
              });
            } else {
              nextBeamsAccumulator.push({
                tokens: newTokens,
                score: newScore,
                completed: false,
              });
            }
          }
        }

        // ビームをスコアでソートし、上位numBeams個を保持 (未完了のものから優先)
        nextBeamsAccumulator.sort((a, b) => b.score - a.score); // スコアで降順ソート
        beams = nextBeamsAccumulator
          .filter((b) => !b.completed)
          .slice(0, numBeams);

        // 完了したシーケンスも保持しつつ、ビーム数を超えないように調整
        // completedSequencesもスコアでソートし、上位を保持するなどの戦略も考えられる

        if (onProgress && step % 5 === 0) {
          const progressInfo: StatusMessage = {
            type: "info",
            message: `推論中 (ステップ ${
              step + 1
            }/${maxLength}, 有効ビーム数: ${beams.length}, 完了シーケンス数: ${
              completedSequences.length
            })...`,
          };
          console.log(`[TranslationSystem][translate] ${progressInfo.message}`);
          onProgress(progressInfo);
        }
      }
      console.log(
        `[TranslationSystem][translate] Beam Search終了 (ステップ完了). 有効ビーム数: ${beams.length}, 完了シーケンス数: ${completedSequences.length}`
      );

      // 最もスコアの高いシーケンスを選択 (完了シーケンスを優先)
      const allCandidates = [
        ...completedSequences,
        ...beams.filter((b) => !b.completed),
      ];
      allCandidates.sort((a, b) => b.score - a.score); // スコアで降順ソート

      const bestSequence = allCandidates[0]?.tokens;

      if (!bestSequence) {
        console.error(
          "[TranslationSystem][translate] 翻訳結果が生成されませんでした。候補シーケンス:",
          allCandidates
        );
        throw new Error("翻訳結果が生成されませんでした。");
      }
      console.log(
        `[TranslationSystem][translate] 最良シーケンス (スコア: ${allCandidates[0]?.score.toFixed(
          4
        )}): ${bestSequence}`
      );

      if (onProgress) {
        const decodeMsg: StatusMessage = {
          type: "info",
          message: "トークンをデコード中...",
        };
        console.log(`[TranslationSystem][translate] ${decodeMsg.message}`);
        onProgress(decodeMsg);
      }
      const translatedText = this.tokenizer.decode(bestSequence, true);
      console.log(
        `[TranslationSystem][translate] 翻訳結果: "${translatedText}"`
      );

      if (onProgress) {
        const successMsg: StatusMessage = {
          type: "success",
          message: "翻訳処理が完了しました。",
        };
        console.log(`[TranslationSystem][translate] ${successMsg.message}`);
        onProgress(successMsg);
      }
      return translatedText;
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      console.error(
        "[TranslationSystem][translate] 翻訳エラー:",
        errorMessage,
        error
      );
      if (onProgress) {
        const errorMsg: StatusMessage = {
          type: "error",
          message: `翻訳に失敗しました: ${errorMessage}`,
        };
        onProgress(errorMsg);
      }
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
    if (array.length === 0) {
      return new Float32Array(0);
    }

    let maxLogit = array[0];
    for (let i = 1; i < array.length; i++) {
      if (array[i] > maxLogit) {
        maxLogit = array[i];
      }
    }

    const result = new Float32Array(array.length);
    let sumExps = 0;

    for (let i = 0; i < array.length; i++) {
      const expVal = Math.exp(array[i] - maxLogit);
      result[i] = expVal;
      sumExps += expVal;
    }

    // ゼロ除算やNaNを防ぐ
    if (sumExps === 0 || !isFinite(sumExps)) {
      // 全ての要素が同じ確率を持つようにフォールバック (あるいは他の戦略)
      const fallbackValue = 1 / array.length;
      for (let i = 0; i < result.length; i++) {
        result[i] = fallbackValue;
      }
      console.warn(
        "[TranslationSystem][softmax] Softmax sumExps is zero or not finite. Applied fallback distribution. Input array sample:",
        array.slice(0, 5)
      );
      return result;
    }

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
      // uint8Array が指す範囲のデータをコピーして新しい ArrayBuffer を作成する
      // これにより、元のバッファが SharedArrayBuffer であっても問題なく ArrayBuffer を取得できる
      const newArrayBuffer = new ArrayBuffer(uint8Array.byteLength);
      new Uint8Array(newArrayBuffer).set(
        new Uint8Array(
          uint8Array.buffer,
          uint8Array.byteOffset,
          uint8Array.byteLength
        )
      );
      return newArrayBuffer;
    } catch (error) {
      console.error("tokenizer.json生成エラー:", error);
      throw new Error(`tokenizer.json生成に失敗: ${error}`);
    }
  }
}
