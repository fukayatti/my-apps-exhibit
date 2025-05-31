import { TokenizeResult } from "./types";

// Transformers.jsを動的にインポートするためのインターフェース
interface TransformersModule {
  AutoTokenizer: {
    from_pretrained: (
      modelName: string,
      options?: Record<string, unknown>
    ) => Promise<TokenizerInterface>;
  };
}

// トークナイザーのインターフェース
interface TokenizerInterface {
  encode: (text: string) => number[];
  decode: (
    tokenIds: number[],
    options?: {
      skip_special_tokens?: boolean;
      clean_up_tokenization_spaces?: boolean;
    }
  ) => string;
  tokenize?: (text: string) => number[];
  decoderStartToken?: number;
  eosToken?: number;
  padToken?: number;
  unkToken?: number;
  getTokenString?: (id: number) => string | null;
}

export class HuggingFaceTokenizer {
  private tokenizer: TokenizerInterface | null = null;
  private isLoaded = false;
  private transformers: TransformersModule | null = null;

  async loadFromPretrained(modelPath: string): Promise<void> {
    try {
      console.log("HuggingFace tokenizerを読み込み中...");

      // Transformers.jsを動的にインポート
      this.transformers = (await import(
        "@xenova/transformers"
      )) as TransformersModule;

      // AutoTokenizerを使用
      this.tokenizer = await this.transformers.AutoTokenizer.from_pretrained(
        modelPath,
        {
          cache_dir: "/.cache/huggingface",
          local_files_only: false,
        }
      );

      this.isLoaded = true;
      console.log("✓ HuggingFace tokenizerの読み込み完了");
    } catch (error) {
      console.error("Tokenizer読み込みエラー:", error);
      throw new Error(`Tokenizerの読み込みに失敗: ${error}`);
    }
  }

  async loadFromBuffer(tokenizerBuffer: ArrayBuffer): Promise<void> {
    try {
      console.log("tokenizer.jsonから読み込み中...");

      // Transformers.jsを動的にインポート
      this.transformers = (await import(
        "@xenova/transformers"
      )) as TransformersModule;

      // tokenizer.jsonの内容を解析
      const tokenizerConfig = JSON.parse(
        new TextDecoder().decode(tokenizerBuffer)
      );

      // 簡易的なBPEトークナイザーを実装
      this.tokenizer = new SimpleBPETokenizer(tokenizerConfig);
      this.isLoaded = true;
      console.log("✓ tokenizer.jsonから読み込み完了");
    } catch (error) {
      console.error("Tokenizer読み込みエラー:", error);
      throw new Error(`tokenizer.jsonの読み込みに失敗: ${error}`);
    }
  }

  async loadFromHuggingFaceFiles(
    vocabBuffer: ArrayBuffer,
    spModelBuffer: ArrayBuffer,
    tokenizerConfigBuffer: ArrayBuffer
  ): Promise<void> {
    try {
      console.log("HuggingFaceファイルから読み込み中...");

      // vocab.jsonを解析
      const vocab = JSON.parse(new TextDecoder().decode(vocabBuffer));

      // tokenizer_config.jsonを解析
      const tokenizerConfig = JSON.parse(
        new TextDecoder().decode(tokenizerConfigBuffer)
      );

      // SentencePieceモデルをHuggingFaceSentencePieceTokenizerで処理
      this.tokenizer = new HuggingFaceSentencePieceTokenizer(
        vocab,
        spModelBuffer,
        tokenizerConfig
      );

      this.isLoaded = true;
      console.log("✓ HuggingFaceファイルから読み込み完了");
    } catch (error) {
      console.error("Tokenizer読み込みエラー:", error);
      throw new Error(`HuggingFaceファイルの読み込みに失敗: ${error}`);
    }
  }

  tokenize(text: string): TokenizeResult {
    if (!this.isLoaded || !this.tokenizer) {
      throw new Error("Tokenizerが読み込まれていません");
    }

    try {
      let encoded: number[];

      if (this.tokenizer.encode) {
        // Transformers.jsのtokenizer
        encoded = this.tokenizer.encode(text);
      } else if (this.tokenizer.tokenize) {
        // SimpleBPETokenizer または HuggingFaceSentencePieceTokenizer
        encoded = this.tokenizer.tokenize(text);
      } else {
        throw new Error("トークナイザーメソッドが見つかりません");
      }

      return {
        input_ids: [Array.isArray(encoded) ? encoded : [encoded]],
      };
    } catch (error) {
      console.error("トークン化エラー:", error);
      throw new Error(`トークン化に失敗: ${error}`);
    }
  }

  decode(tokenIds: number[], skipSpecialTokens = true): string {
    if (!this.isLoaded || !this.tokenizer) {
      throw new Error("Tokenizerが読み込まれていません");
    }

    try {
      if (this.tokenizer.decode) {
        // Transformers.jsのtokenizer または カスタムtokenizer
        return this.tokenizer.decode(tokenIds, {
          skip_special_tokens: skipSpecialTokens,
        });
      }
      return "";
    } catch (error) {
      console.error("デコードエラー:", error);
      throw new Error(`デコードに失敗: ${error}`);
    }
  }

  get decoderStartToken(): number | null {
    if (!this.tokenizer) return null;
    return this.tokenizer.decoderStartToken || 2;
  }

  get eosToken(): number | null {
    if (!this.tokenizer) return null;
    return this.tokenizer.eosToken || 1;
  }

  get padToken(): number | null {
    if (!this.tokenizer) return null;
    return this.tokenizer.padToken || 0;
  }

  get unkToken(): number | null {
    if (!this.tokenizer) return null;
    return this.tokenizer.unkToken || 3;
  }

  setDecoderStartTokenId(id: number): void {
    if (this.tokenizer) {
      this.tokenizer.decoderStartToken = id;
    }
  }

  setEosTokenId(id: number): void {
    if (this.tokenizer) {
      this.tokenizer.eosToken = id;
    }
  }

  getTokenString(id: number): string | null {
    if (!this.tokenizer) return null;

    try {
      if (this.tokenizer.getTokenString) {
        return this.tokenizer.getTokenString(id);
      }
      return this.tokenizer.decode([id]);
    } catch {
      return null;
    }
  }

  getLangToken(langCode: string): string {
    return `__${langCode}__`;
  }

  getLangId(langCode: string): number | undefined {
    if (!this.tokenizer) return undefined;

    const langToken = this.getLangToken(langCode);
    if (this.tokenizer.encode) {
      const encoded = this.tokenizer.encode(langToken);
      return Array.isArray(encoded) ? encoded[0] : encoded;
    }
    return undefined;
  }
}

// 簡易BPEトークナイザーの実装
class SimpleBPETokenizer implements TokenizerInterface {
  private vocab: Record<string, number> = {};
  private decoder: Record<number, string> = {};
  private merges: string[] = [];
  private specialTokens: Record<string, number> = {};

  constructor(config: Record<string, unknown>) {
    if (
      config.model &&
      typeof config.model === "object" &&
      config.model !== null
    ) {
      const model = config.model as Record<string, unknown>;
      if (model.vocab && typeof model.vocab === "object") {
        this.vocab = model.vocab as Record<string, number>;

        // decoderを作成
        for (const [token, id] of Object.entries(this.vocab)) {
          this.decoder[id as number] = token;
        }
      }

      if (model.merges && Array.isArray(model.merges)) {
        this.merges = model.merges as string[];
      }
    }

    // 特殊トークンの設定
    if (config.added_tokens && Array.isArray(config.added_tokens)) {
      for (const tokenInfo of config.added_tokens) {
        if (typeof tokenInfo === "object" && tokenInfo !== null) {
          const token = tokenInfo as Record<string, unknown>;
          if (
            typeof token.content === "string" &&
            typeof token.id === "number"
          ) {
            this.specialTokens[token.content] = token.id;
          }
        }
      }
    }

    // デフォルトの特殊トークン
    this.specialTokens["<pad>"] = this.specialTokens["<pad>"] || 0;
    this.specialTokens["</s>"] = this.specialTokens["</s>"] || 1;
    this.specialTokens["<s>"] = this.specialTokens["<s>"] || 2;
    this.specialTokens["<unk>"] = this.specialTokens["<unk>"] || 3;
  }

  encode(text: string): number[] {
    return this.tokenize(text);
  }

  tokenize(text: string): number[] {
    // 簡易的なBPE実装
    const tokens = this.bpeEncode(text);
    return tokens.map(
      (token) => this.vocab[token] || this.specialTokens["<unk>"] || 3
    );
  }

  private bpeEncode(text: string): string[] {
    // 基本的な前処理
    const normalizedText = text.trim();
    if (!normalizedText) return [];

    // 文字レベルで開始
    const tokens = normalizedText
      .split("")
      .map((char) => (char === " " ? "▁" : char));

    // マージルールを適用
    for (const merge of this.merges) {
      const [first, second] = merge.split(" ");
      if (!first || !second) continue;

      const newTokens: string[] = [];
      let i = 0;

      while (i < tokens.length) {
        if (
          i < tokens.length - 1 &&
          tokens[i] === first &&
          tokens[i + 1] === second
        ) {
          newTokens.push(first + second);
          i += 2;
        } else {
          newTokens.push(tokens[i]);
          i++;
        }
      }

      tokens.splice(0, tokens.length, ...newTokens);
    }

    return tokens;
  }

  decode(
    tokenIds: number[],
    options?: { skip_special_tokens?: boolean }
  ): string {
    const skipSpecialTokens = options?.skip_special_tokens ?? true;
    const tokens = tokenIds
      .map((id) => {
        const token = this.decoder[id];
        if (!token) return "";

        if (
          skipSpecialTokens &&
          Object.values(this.specialTokens).includes(id)
        ) {
          return "";
        }

        return token;
      })
      .filter((token) => token !== "");

    return tokens.join("").replace(/▁/g, " ").trim();
  }

  get decoderStartToken(): number {
    return this.specialTokens["<s>"] || 2;
  }

  get eosToken(): number {
    return this.specialTokens["</s>"] || 1;
  }

  get padToken(): number {
    return this.specialTokens["<pad>"] || 0;
  }

  get unkToken(): number {
    return this.specialTokens["<unk>"] || 3;
  }

  getTokenString(id: number): string | null {
    return this.decoder[id] || null;
  }
}

// HuggingFace形式のSentencePieceトークナイザーの実装
class HuggingFaceSentencePieceTokenizer implements TokenizerInterface {
  private vocab: Record<string, number> = {};
  private decoder: Record<number, string> = {};
  private specialTokens: Record<string, number> = {};
  private spModelBuffer: ArrayBuffer;
  private config: Record<string, unknown>;

  constructor(
    vocab: Record<string, number>,
    spModelBuffer: ArrayBuffer,
    config: Record<string, unknown>
  ) {
    this.vocab = vocab;
    this.spModelBuffer = spModelBuffer;
    this.config = config;

    // decoderを作成
    for (const [token, id] of Object.entries(this.vocab)) {
      this.decoder[id as number] = token;
    }

    // 特殊トークンの設定
    this.setupSpecialTokens();
  }

  private setupSpecialTokens(): void {
    // configから特殊トークンを取得
    const specialTokensMap = (this.config.special_tokens_map || {}) as Record<
      string,
      unknown
    >;

    // よく使われる特殊トークン
    this.specialTokens["<pad>"] = this.vocab["<pad>"] || 0;
    this.specialTokens["</s>"] = this.vocab["</s>"] || 1;
    this.specialTokens["<s>"] = this.vocab["<s>"] || 2;
    this.specialTokens["<unk>"] = this.vocab["<unk>"] || 3;

    // configで定義された特殊トークンを追加
    for (const [, tokenValue] of Object.entries(specialTokensMap)) {
      if (
        typeof tokenValue === "string" &&
        this.vocab[tokenValue] !== undefined
      ) {
        this.specialTokens[tokenValue] = this.vocab[tokenValue];
      }
    }
  }

  encode(text: string): number[] {
    return this.tokenize(text);
  }

  tokenize(text: string): number[] {
    // 簡易的なSentencePiece実装
    const tokens = this.sentencePieceEncode(text);
    return tokens.map(
      (token) => this.vocab[token] || this.specialTokens["<unk>"] || 3
    );
  }

  private sentencePieceEncode(text: string): string[] {
    // 基本的な前処理
    const normalizedText = text.trim();
    if (!normalizedText) return [];

    // スペースを▁に置換（SentencePieceの標準）
    const preprocessed = "▁" + normalizedText.replace(/ /g, "▁");

    // 文字レベルで開始
    const tokens = preprocessed.split("");

    // 語彙にある最長マッチを探す
    const result: string[] = [];
    let i = 0;

    while (i < tokens.length) {
      let matched = false;

      // 最長マッチを探す（最大10文字まで）
      for (let len = Math.min(tokens.length - i, 10); len >= 1; len--) {
        const candidate = tokens.slice(i, i + len).join("");
        if (this.vocab[candidate] !== undefined) {
          result.push(candidate);
          i += len;
          matched = true;
          break;
        }
      }

      if (!matched) {
        // 単一文字もなければ<unk>
        const char = tokens[i];
        if (this.vocab[char] !== undefined) {
          result.push(char);
        } else {
          result.push("<unk>");
        }
        i++;
      }
    }

    return result;
  }

  decode(
    tokenIds: number[],
    options?: { skip_special_tokens?: boolean }
  ): string {
    const skipSpecialTokens = options?.skip_special_tokens ?? true;
    const tokens = tokenIds
      .map((id) => {
        const token = this.decoder[id];
        if (!token) return "";

        if (
          skipSpecialTokens &&
          Object.values(this.specialTokens).includes(id)
        ) {
          return "";
        }

        return token;
      })
      .filter((token) => token !== "");

    // SentencePieceの▁をスペースに戻す
    return tokens.join("").replace(/▁/g, " ").trim();
  }

  get decoderStartToken(): number {
    return this.specialTokens["<s>"] || 2;
  }

  get eosToken(): number {
    return this.specialTokens["</s>"] || 1;
  }

  get padToken(): number {
    return this.specialTokens["<pad>"] || 0;
  }

  get unkToken(): number {
    return this.specialTokens["<unk>"] || 3;
  }

  getTokenString(id: number): string | null {
    return this.decoder[id] || null;
  }
}

// 後方互換性のためのエイリアス
export const SMALL100TokenizerJS = HuggingFaceTokenizer;
