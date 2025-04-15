import { Hono, Context } from "hono";
import { handle } from "hono/vercel";
import { prettyJSON } from "hono/pretty-json";
import { load } from "cheerio";

export const runtime = "edge";

// 全角文字を半角に変換する
function fullwidthToHalfwidth(str: string): string {
  return str
    .split("")
    .map((c) => {
      if (c === "\u3000") return " ";
      const code = c.charCodeAt(0);
      if (code >= 0xff01 && code <= 0xff5e) {
        return String.fromCharCode(code - 0xfee0);
      }
      return c;
    })
    .join("");
}

// 改行・連続スペースを1スペースに統一する
function normalizeWhitespace(str: string): string {
  return str.replace(/\s+/g, " ").trim();
}

// シンボルのマッピング：◉→休講、◎→補講、◇→遠隔授業、☆→授業・教室変更
function symbolMap(symbol: string): string {
  switch (symbol) {
    case "◉":
      return "休講";
    case "◎":
      return "補講";
    case "◇":
      return "遠隔授業";
    case "☆":
      return "授業・教室変更";
    default:
      return "";
  }
}

// エントリー行解析用正規表現
// 例: "☆専攻科１年　５・６限　応用制御工学（平澤）⇒７・８限へ"
const entryRegex =
  /^(?<symbol>[◉◎◇☆])\s*(?<class>[^\s]+)\s+(?<period>\d+・\d+限)\s+(?<rest>.+)$/;
// "⇒" 後が「数字・数字限」または「数字・数字限へ」かを判定する正規表現
const periodRegex = /^(?<period2>\d+・\d+限)(?:へ)?$/;

// 型定義
interface Entry {
  type: string;
  class: string;
  period: string;
  subject1: string | null;
  subject2: string | null;
  period2: string | null;
}

interface DateEntry {
  date: string;
  entries: Entry[];
}

interface FinalOutput {
  休講情報: DateEntry[];
}

const app = new Hono();

// prettyJSON ミドルウェアを全ルートに適用（レスポンス JSON を整形して返す）
app.use("*", prettyJSON());

// GET "/" へのルート処理
app.get("/", async (c: Context) => {
  try {
    const url = "https://www.ibaraki-ct.ac.jp/info/archives/65544";
    const response = await fetch(url);
    const html: string = await response.text();

    const $ = load(html);
    // 休講情報本文が含まれる領域は id="post_main" と仮定
    const $postMain = $("#post_main");
    if (!$postMain.length) {
      return c.json({ error: "post_main が見つかりません" }, 404);
    }

    // 結果を日付ごとに格納するオブジェクト（キー: 日付文字列）
    const results: { [date: string]: Entry[] } = {};
    let currentDate = "";

    // $postMain 内の <p> タグごとに走査
    $postMain.find("p").each((_, elem) => {
      const $p = $(elem);
      let text: string = $p.text();
      text = normalizeWhitespace(fullwidthToHalfwidth(text));

      // <mark> タグがあれば、この段落は日付情報とみなす
      if ($p.find("mark").length > 0) {
        currentDate = text;
        if (!results[currentDate]) {
          results[currentDate] = [];
        }
      } else {
        // 日付が未設定ならスキップ
        if (!currentDate) return;
        const match = entryRegex.exec(text);
        if (match && match.groups) {
          const { symbol, class: classText, period, rest } = match.groups;
          const typeField = symbolMap(symbol);
          let subject1: string | null = null;
          let subject2: string | null = null;
          let period2: string | null = null;
          if (rest.includes("⇒")) {
            const parts = rest.split("⇒", 2).map((s) => s.trim());
            subject1 = parts[0] === "授業なし" ? null : parts[0];
            const periodMatch = periodRegex.exec(parts[1]);
            if (
              periodMatch &&
              periodMatch.groups &&
              periodMatch.groups.period2
            ) {
              period2 = periodMatch.groups.period2.trim();
              subject2 = null;
            } else {
              subject2 = parts[1] === "授業なし" ? null : parts[1];
            }
          } else {
            subject1 = rest === "授業なし" ? null : rest;
          }
          const entry: Entry = {
            type: typeField,
            class: classText.trim(),
            period: period.trim(),
            subject1,
            subject2,
            period2,
          };
          if (!results[currentDate]) {
            results[currentDate] = [];
          }
          results[currentDate].push(entry);
        }
      }
    });

    const finalOutput: FinalOutput = {
      休講情報: Object.keys(results)
        .sort()
        .map((date) => ({
          date,
          entries: results[date],
        })),
    };

    return c.json(finalOutput);
  } catch (error: any) {
    console.error("Error:", error);
    return c.json({ error: error.message }, 500);
  }
});

// Next.js の API ルートとして Hono アプリケーションをエッジ関数で提供
export const GET = handle(app);
