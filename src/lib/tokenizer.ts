import { TokenizeResult } from "./types";
import { ModelDownloader } from "./model-downloader";

// 言語コードの定義
const FAIRSEQ_LANGUAGE_CODES = {
  m2m100: [
    "af",
    "am",
    "ar",
    "ast",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "br",
    "bs",
    "ca",
    "ceb",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fa",
    "ff",
    "fi",
    "fr",
    "fy",
    "ga",
    "gd",
    "gl",
    "gu",
    "ha",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "ig",
    "ilo",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "lb",
    "lg",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "my",
    "ne",
    "nl",
    "no",
    "ns",
    "oc",
    "or",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sd",
    "si",
    "sk",
    "sl",
    "so",
    "sq",
    "sr",
    "ss",
    "su",
    "sv",
    "sw",
    "ta",
    "th",
    "tl",
    "tn",
    "tr",
    "uk",
    "ur",
    "uz",
    "vi",
    "wo",
    "xh",
    "yi",
    "yo",
    "zh",
    "zu",
  ],
};

const SPIECE_UNDERLINE = "▁";

// SentencePieceProcessor の簡易実装
class SimpleSentencePieceProcessor {
  private vocab: Map<string, number> = new Map();
  private idToToken: Map<number, string> = new Map();
  private pieces: Array<{ piece: string; score: number; type: number }> = [];

  constructor() {}

  async load(_modelData: ArrayBuffer): Promise<void> {
    // SentencePieceモデルファイルを解析する
    // 実際の実装では、プロトバッファの解析が必要だが、
    // ここでは簡略化してvocab.jsonから構築する
    console.log("SentencePieceモデルを読み込み中...");
  }

  loadFromVocab(vocab: Record<string, number>): void {
    // vocab.jsonからSentencePieceの語彙を構築
    for (const [piece, id] of Object.entries(vocab)) {
      this.vocab.set(piece, id);
      this.idToToken.set(id, piece);
      this.pieces.push({ piece, score: 0, type: 1 });
    }
  }

  encode(text: string, outType: "str" | "id" = "id"): string[] | number[] {
    // テキストを正規化
    const normalizedText = this.normalize(text);

    // BPEスタイルのトークン化
    const tokens = this.tokenize(normalizedText);

    if (outType === "str") {
      return tokens;
    } else {
      return tokens.map(
        (token) => this.vocab.get(token) || this.vocab.get("<unk>") || 3
      );
    }
  }

  decode(tokens: string[] | number[]): string {
    let pieces: string[];

    if (typeof tokens[0] === "number") {
      pieces = (tokens as number[]).map(
        (id) => this.idToToken.get(id) || "<unk>"
      );
    } else {
      pieces = tokens as string[];
    }

    // SentencePieceの復号化
    return pieces
      .join("")
      .replace(new RegExp(SPIECE_UNDERLINE, "g"), " ")
      .trim();
  }

  private normalize(text: string): string {
    // スペースをSentencePieceのunderlineに変換
    return SPIECE_UNDERLINE + text.replace(/ /g, SPIECE_UNDERLINE);
  }

  private tokenize(text: string): string[] {
    const result: string[] = [];
    let i = 0;

    while (i < text.length) {
      let found = false;

      // 最長マッチを探す
      for (let len = Math.min(text.length - i, 20); len >= 1; len--) {
        const candidate = text.slice(i, i + len);
        if (this.vocab.has(candidate)) {
          result.push(candidate);
          i += len;
          found = true;
          break;
        }
      }

      if (!found) {
        // 単一文字で試す
        const char = text[i];
        if (this.vocab.has(char)) {
          result.push(char);
        } else {
          result.push("<unk>");
        }
        i++;
      }
    }

    return result;
  }
}

// SMALL100Tokenizer のTypeScript実装
export class SMALL100Tokenizer {
  private encoder: Record<string, number> = {};
  private decoder: Record<number, string> = {};
  private spModel: SimpleSentencePieceProcessor | null = null;
  private langCodeToToken: Record<string, string> = {};
  private langTokenToId: Record<string, number> = {};
  private langCodeToId: Record<string, number> = {};
  private idToLangToken: Record<number, string> = {};
  private encoderSize = 0;
  private _tgtLang = "en";
  private curLangId = 0;
  private prefixTokens: number[] = [];
  private suffixTokens: number[] = [];
  private numMadeupWords = 8;

  // 特殊トークンのデフォルト値
  private bosTokenId = 0; // <s>
  private eosTokenId = 1; // </s>
  private padTokenId = 2; // <pad>
  private unkTokenId = 3; // <unk>

  private specialTokens: Record<string, number> = {
    "<s>": 0,
    "</s>": 1,
    "<pad>": 2,
    "<unk>": 3,
  };

  constructor() {
    // 言語コードの初期化
    const fairseqLanguageCodes = FAIRSEQ_LANGUAGE_CODES["m2m100"];
    this.langCodeToToken = {};
    for (const langCode of fairseqLanguageCodes) {
      this.langCodeToToken[langCode] = `__${langCode}__`;
    }
  }

  async loadFromPretrained(
    repoId = "alirezamsh/small100",
    tgtLang = "en"
  ): Promise<void> {
    try {
      console.log(`SMALL100Tokenizer を ${repoId} から読み込み中...`);

      const downloader = new ModelDownloader(
        `https://huggingface.co/${repoId}/resolve/main/`,
        [
          { name: "vocab.json", size: "unknown" },
          { name: "sentencepiece.bpe.model", size: "unknown" },
          { name: "tokenizer_config.json", size: "unknown" },
        ]
      );

      // vocab.jsonを読み込み
      const vocabBuffer = await downloader.downloadFile(
        "vocab.json",
        (loaded, total) => {
          const progress = total ? (loaded / total) * 100 : 0;
          console.log(`vocab.json ダウンロード中: ${progress.toFixed(2)}%`);
        }
      );

      if (!vocabBuffer) {
        throw new Error("vocab.json のダウンロードに失敗しました。");
      }

      // SentencePieceモデルを読み込み
      const spModelBuffer = await downloader.downloadFile(
        "sentencepiece.bpe.model",
        (loaded, total) => {
          const progress = total ? (loaded / total) * 100 : 0;
          console.log(
            `sentencepiece.bpe.model ダウンロード中: ${progress.toFixed(2)}%`
          );
        }
      );

      if (!spModelBuffer) {
        throw new Error(
          "sentencepiece.bpe.model のダウンロードに失敗しました。"
        );
      }

      // tokenizer_config.jsonを読み込み
      const tokenizerConfigBuffer = await downloader.downloadFile(
        "tokenizer_config.json",
        (loaded, total) => {
          const progress = total ? (loaded / total) * 100 : 0;
          console.log(
            `tokenizer_config.json ダウンロード中: ${progress.toFixed(2)}%`
          );
        }
      );

      // 設定を初期化
      await this.initializeFromBuffers(
        vocabBuffer,
        spModelBuffer,
        tokenizerConfigBuffer,
        tgtLang
      );

      console.log("✓ SMALL100Tokenizer の読み込み完了");
    } catch (error) {
      console.error("SMALL100Tokenizer読み込みエラー:", error);
      throw new Error(`SMALL100Tokenizerの読み込みに失敗: ${error}`);
    }
  }

  async loadFromBuffers(
    vocabBuffer: ArrayBuffer,
    spModelBuffer: ArrayBuffer,
    tokenizerConfigBuffer?: ArrayBuffer,
    tgtLang = "en"
  ): Promise<void> {
    await this.initializeFromBuffers(
      vocabBuffer,
      spModelBuffer,
      tokenizerConfigBuffer,
      tgtLang
    );
  }

  private async initializeFromBuffers(
    vocabBuffer: ArrayBuffer,
    spModelBuffer: ArrayBuffer,
    tokenizerConfigBuffer?: ArrayBuffer,
    tgtLang = "en"
  ): Promise<void> {
    // vocab.jsonを解析
    this.encoder = JSON.parse(new TextDecoder().decode(vocabBuffer));
    this.decoder = {};
    for (const [token, id] of Object.entries(this.encoder)) {
      this.decoder[id as number] = token;
    }

    this.encoderSize = Object.keys(this.encoder).length;

    // 特殊トークンの設定
    if (tokenizerConfigBuffer) {
      const tokenizerConfig = JSON.parse(
        new TextDecoder().decode(tokenizerConfigBuffer)
      );
      this.updateSpecialTokensFromConfig(tokenizerConfig);
    }

    // SentencePieceプロセッサーを初期化
    this.spModel = new SimpleSentencePieceProcessor();
    await this.spModel.load(spModelBuffer);
    this.spModel.loadFromVocab(this.encoder);

    // 言語トークンのマッピングを構築
    this.buildLanguageTokenMappings();

    // ターゲット言語を設定
    this._tgtLang = tgtLang;
    this.setLangSpecialTokens(this._tgtLang);
  }

  private updateSpecialTokensFromConfig(config: Record<string, unknown>): void {
    if (config.bos_token && typeof config.bos_token === "string") {
      const bosId = this.encoder[config.bos_token];
      if (bosId !== undefined) {
        this.bosTokenId = bosId;
        this.specialTokens["<s>"] = bosId;
      }
    }

    if (config.eos_token && typeof config.eos_token === "string") {
      const eosId = this.encoder[config.eos_token];
      if (eosId !== undefined) {
        this.eosTokenId = eosId;
        this.specialTokens["</s>"] = eosId;
      }
    }

    if (config.pad_token && typeof config.pad_token === "string") {
      const padId = this.encoder[config.pad_token];
      if (padId !== undefined) {
        this.padTokenId = padId;
        this.specialTokens["<pad>"] = padId;
      }
    }

    if (config.unk_token && typeof config.unk_token === "string") {
      const unkId = this.encoder[config.unk_token];
      if (unkId !== undefined) {
        this.unkTokenId = unkId;
        this.specialTokens["<unk>"] = unkId;
      }
    }
  }

  private buildLanguageTokenMappings(): void {
    const fairseqLanguageCodes = FAIRSEQ_LANGUAGE_CODES["m2m100"];

    this.langTokenToId = {};
    this.langCodeToId = {};
    this.idToLangToken = {};

    for (let i = 0; i < fairseqLanguageCodes.length; i++) {
      const langCode = fairseqLanguageCodes[i];
      const langToken = this.getLangToken(langCode);
      const langId = this.encoderSize + i;

      this.langTokenToId[langToken] = langId;
      this.langCodeToId[langCode] = langId;
      this.idToLangToken[langId] = langToken;
    }
  }

  get vocabSize(): number {
    return (
      this.encoderSize +
      Object.keys(this.langTokenToId).length +
      this.numMadeupWords
    );
  }

  get tgtLang(): string {
    return this._tgtLang;
  }

  set tgtLang(newTgtLang: string) {
    this._tgtLang = newTgtLang;
    this.setLangSpecialTokens(this._tgtLang);
  }

  tokenize(text: string): TokenizeResult {
    if (!this.spModel) {
      throw new Error("SentencePieceモデルが初期化されていません");
    }

    try {
      const tokens = this.spModel.encode(text, "str") as string[];
      const tokenIds = tokens.map((token) => this.convertTokenToId(token));

      return {
        input_ids: [tokenIds],
      };
    } catch (error) {
      console.error("トークン化エラー:", error);
      throw new Error(`トークン化に失敗: ${error}`);
    }
  }

  encode(text: string): number[] {
    if (!this.spModel) {
      throw new Error("SentencePieceモデルが初期化されていません");
    }

    const tokens = this.spModel.encode(text, "str") as string[];
    return tokens.map((token) => this.convertTokenToId(token));
  }

  private convertTokenToId(token: string): number {
    // 言語トークンのチェック
    if (this.langTokenToId[token] !== undefined) {
      return this.langTokenToId[token];
    }

    // 通常の語彙からの変換
    return this.encoder[token] || this.unkTokenId;
  }

  private convertIdToToken(index: number): string {
    // 言語トークンのチェック
    if (this.idToLangToken[index]) {
      return this.idToLangToken[index];
    }

    // 通常の語彙からの変換
    return this.decoder[index] || "<unk>";
  }

  decode(tokenIds: number[], skipSpecialTokens = true): string {
    if (!this.spModel) {
      throw new Error("SentencePieceモデルが初期化されていません");
    }

    try {
      const tokens = tokenIds
        .map((id) => {
          const token = this.convertIdToToken(id);

          // 特殊トークンをスキップするかチェック
          if (skipSpecialTokens && this.isSpecialToken(token)) {
            return "";
          }

          return token;
        })
        .filter((token) => token !== "");

      return this.spModel.decode(tokens);
    } catch (error) {
      console.error("デコードエラー:", error);
      throw new Error(`デコードに失敗: ${error}`);
    }
  }

  private isSpecialToken(token: string): boolean {
    return (
      Object.keys(this.specialTokens).includes(token) ||
      (token.startsWith("__") && token.endsWith("__"))
    );
  }

  buildInputsWithSpecialTokens(
    tokenIds0: number[],
    tokenIds1?: number[]
  ): number[] {
    if (tokenIds1 === undefined) {
      if (this.prefixTokens.length === 0) {
        return [...tokenIds0, ...this.suffixTokens];
      } else {
        return [...this.prefixTokens, ...tokenIds0, ...this.suffixTokens];
      }
    }

    // ペアの場合
    if (this.prefixTokens.length === 0) {
      return [...tokenIds0, ...tokenIds1, ...this.suffixTokens];
    } else {
      return [
        ...this.prefixTokens,
        ...tokenIds0,
        ...tokenIds1,
        ...this.suffixTokens,
      ];
    }
  }

  setLangSpecialTokens(srcLang: string): void {
    const langToken = this.getLangToken(srcLang);
    this.curLangId = this.langTokenToId[langToken];
    this.prefixTokens = [this.curLangId];
    this.suffixTokens = [this.eosTokenId];
  }

  getLangToken(lang: string): string {
    return this.langCodeToToken[lang] || `__${lang}__`;
  }

  getLangId(lang: string): number {
    const langToken = this.getLangToken(lang);
    return this.langTokenToId[langToken];
  }

  // 特殊トークンのgetterプロパティ
  get decoderStartToken(): number {
    return this.bosTokenId;
  }

  get eosToken(): number {
    return this.eosTokenId;
  }

  get padToken(): number {
    return this.padTokenId;
  }

  get unkToken(): number {
    return this.unkTokenId;
  }

  setDecoderStartTokenId(id: number): void {
    this.bosTokenId = id;
  }

  setEosTokenId(id: number): void {
    this.eosTokenId = id;
  }

  getTokenString(id: number): string | null {
    return this.convertIdToToken(id);
  }

  // 入力モードと出力モードの切り替え
  switchToInputMode(): void {
    this.setLangSpecialTokens(this._tgtLang);
  }

  switchToTargetMode(): void {
    this.prefixTokens = [];
    this.suffixTokens = [this.eosTokenId];
  }
}

// 後方互換性のためのエイリアス（既存のHuggingFaceTokenizerインターフェースと互換性を保つ）
export class HuggingFaceTokenizer extends SMALL100Tokenizer {
  private isLoaded = false;

  async loadFromPretrained(
    repoId = "alirezamsh/small100",
    _filename = "tokenizer.json" // このパラメータは無視される
  ): Promise<void> {
    await super.loadFromPretrained(repoId);
    this.isLoaded = true;
  }

  async loadFromBuffer(tokenizerBuffer: ArrayBuffer): Promise<void> {
    // 新しい実装では使用しない
    throw new Error(
      "loadFromBufferは新しいSMALL100Tokenizer実装では非サポートです。loadFromBuffersを使用してください。"
    );
  }

  async loadFromHuggingFaceFiles(
    vocabBuffer: ArrayBuffer,
    spModelBuffer: ArrayBuffer,
    tokenizerConfigBuffer: ArrayBuffer
  ): Promise<void> {
    await this.loadFromBuffers(
      vocabBuffer,
      spModelBuffer,
      tokenizerConfigBuffer
    );
    this.isLoaded = true;
  }
}

// 後方互換性のためのエイリアス
export const SMALL100TokenizerJS = SMALL100Tokenizer;
