import * as ort from "onnxruntime-web";
import { ModelStorage } from "./model-storage";
import { ModelDownloader } from "./model-downloader";
import { SMALL100Tokenizer } from "./tokenizer";
import {
  TranslationConfig,
  StatusMessage,
  ModelInfo,
  // TokenizerConfig, // 未使用のため削除
  // AddedToken, // 未使用のため削除
  FileStatus,
} from "./types";

export class TranslationSystem {
  private session: ort.InferenceSession | null = null;
  private tokenizer: SMALL100Tokenizer | null = null;
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
        "sentencepiece.bpe.model",
        "config.json",
        "tokenizer_config.json",
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
      this.tokenizer = new SMALL100Tokenizer();

      const vocabBuffer = files["vocab.json"];
      const spModelBuffer = files["sentencepiece.bpe.model"];
      const tokenizerConfigBuffer = files["tokenizer_config.json"];

      if (!vocabBuffer) {
        throw new Error("vocab.json がファイルキャッシュに見つかりません。");
      }
      if (!spModelBuffer) {
        throw new Error(
          "sentencepiece.bpe.model がファイルキャッシュに見つかりません。"
        );
      }

      console.log("[TranslationSystem] SMALL100Tokenizer を初期化します。");
      await this.tokenizer.loadFromBuffers(
        vocabBuffer,
        spModelBuffer,
        tokenizerConfigBuffer
      );

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
      "sentencepiece.bpe.model",
      "config.json",
      "tokenizer_config.json",
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
          console.log(
            `[TranslationSystem][translate][BeamStep ${step}] Processing beam: tokens=${
              beam.tokens
            }, score=${beam.score.toFixed(4)}`
          );

          feeds.decoder_input_ids = new ort.Tensor(
            "int64",
            BigInt64Array.from(beam.tokens.map(BigInt)),
            [1, beam.tokens.length]
          );
          // console.log(`[TranslationSystem][translate][BeamStep ${step}] Decoder input_ids: ${beam.tokens}`);

          const output = await this.session.run(feeds);
          const logits = output.logits.data as Float32Array;
          // console.log(`[TranslationSystem][translate][BeamStep ${step}] Logits shape: ${output.logits.dims.join(',')}, length: ${logits.length}`);

          // Logitsの最後のトークンに対応する部分のみを取得
          // Logitsの形状は [batch_size, sequence_length, vocab_size]
          // ここでは batch_size = 1, sequence_length = beam.tokens.length
          const vocabSize = this.config.vocab_size;
          const sequenceLength = beam.tokens.length;
          const nextTokenLogits = logits.slice(
            (sequenceLength - 1) * vocabSize,
            sequenceLength * vocabSize
          );
          // console.log(`[TranslationSystem][translate][BeamStep ${step}] Next token logits (slice from ${(sequenceLength - 1) * vocabSize} to ${sequenceLength * vocabSize}, length ${nextTokenLogits.length}):`, nextTokenLogits.slice(0, 10));

          const probabilities = this.softmax(nextTokenLogits);
          // console.log(`[TranslationSystem][translate][BeamStep ${step}] Probabilities (sample):`, probabilities.slice(0, 10));
          const topK = this.getTopK(probabilities, numBeams * 2); // 探索候補を少し増やす (numBeams * 2 程度)
          // console.log(`[TranslationSystem][translate][BeamStep ${step}] Top K tokens:`, topK.map(t => ({id: t.index, prob: t.probability.toFixed(6)})));

          for (const { index: tokenId, probability } of topK) {
            if (probability < 1e-9) continue; // ごくわずかな確率のトークンは無視 (log(0)対策)

            const newTokens = [...beam.tokens, tokenId];
            const newScore = beam.score + Math.log(probability);
            // console.log(`[TranslationSystem][translate][BeamStep ${step}] Candidate: newTokens=${newTokens}, newScore=${newScore.toFixed(4)}, tokenId=${tokenId}, prob=${probability.toFixed(6)}`);

            if (
              tokenId === this.tokenizer.eosToken ||
              newTokens.length >= maxLength
            ) {
              // console.log(`[TranslationSystem][translate][BeamStep ${step}] Completed sequence: tokens=${newTokens}, score=${newScore.toFixed(4)}, EOS=${tokenId === this.tokenizer.eosToken}, MaxLength=${newTokens.length >= maxLength}`);
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
        // console.log(`[TranslationSystem][translate][BeamStep ${step}] Next beams accumulator (before sort/slice):`, nextBeamsAccumulator.map(b => ({tokens: b.tokens, score:b.score.toFixed(4), completed: b.completed })));

        // ビームをスコアでソートし、上位numBeams個を保持 (未完了のものから優先)
        nextBeamsAccumulator.sort((a, b) => b.score - a.score);

        const activeBeams = nextBeamsAccumulator.filter((b) => !b.completed);
        const newCompleted = nextBeamsAccumulator.filter((b) => b.completed);

        // 新しく完了したシーケンスを追加 (重複はスコアで考慮されるはず)
        completedSequences.push(...newCompleted);
        // スコアでソートし、上位 N 個だけ保持するなどの枝刈りも有効
        completedSequences.sort((a, b) => b.score - a.score);
        // if (completedSequences.length > numBeams * 2) { // あまり多くなりすぎないように
        //   completedSequences.splice(numBeams * 2);
        // }

        beams = activeBeams.slice(0, numBeams);
        // console.log(`[TranslationSystem][translate][BeamStep ${step}] Updated beams:`, beams.map(b => ({tokens: b.tokens, score:b.score.toFixed(4)})));
        // console.log(`[TranslationSystem][translate][BeamStep ${step}] Updated completedSequences (count: ${completedSequences.length}):`, completedSequences.slice(0, numBeams).map(s => ({tokens: s.tokens, score: s.score.toFixed(4)})));

        // 完了したシーケンスも保持しつつ、ビーム数を超えないように調整
        // completedSequencesもスコアでソートし、上位を保持するなどの戦略も考えられる

        if (onProgress && (step % 5 === 0 || step === maxLength - 1)) {
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

  // createTokenizerJson メソッドは tokenizer.json を直接使用するため不要になった
  // private createTokenizerJson(
  //   vocabBuffer: ArrayBuffer,
  //   tokenizerConfigBuffer: ArrayBuffer
  // ): ArrayBuffer {
  //   // ... (以前の実装)
  // }
}
