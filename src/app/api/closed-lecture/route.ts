// src/app/api/closed-lecture/route.ts
import { Hono, Context } from "hono";
import { prettyJSON } from "hono/pretty-json";
import { load } from "cheerio";

// 指定オフセット日数の日付文字列を手動生成 (例: "4/16(水)")
function getDateStr(offsetDays: number): string {
  const d = new Date();
  d.setDate(d.getDate() + offsetDays);
  const weekdays = ["日", "月", "火", "水", "木", "金", "土"];
  const m = d.getMonth() + 1;
  const day = d.getDate();
  const wd = weekdays[d.getDay()];
  return `${m}/${day}(${wd})`;
}

// URLデコード→全角→半角変換
function fullwidthToHalfwidth(str: string): string {
  try {
    str = decodeURIComponent(str);
  } catch {
    // 既にデコード済みの場合
  }
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

// 空白正規化
function normalizeWhitespace(str: string): string {
  return str.replace(/\s+/g, " ").trim();
}

// 記号マッピング
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

const entryRegex =
  /^(?<symbol>[◉◎◇☆])\s*(?<cls>[^\s]+)\s+(?<period>\d+・\d+限)\s+(?<rest>.+)$/;
const periodRegex = /^(?<period2>\d+・\d+限)(?:へ)?$/;

type Entry = {
  type: string;
  class: string;
  period1: string;
  period2: string;
  subject1: string | null;
  subject2: string | null;
};

type DateEntry = {
  date: string;
  entries: Entry[];
};

type FinalOutput = { 休講情報: DateEntry[] };

const app = new Hono();
app.use("*", prettyJSON());

app.get("/api/closed-lecture", async (c: Context) => {
  // オフセット日数 (0なら今日, 1なら明日など)
  const offsetRaw = c.req.query("offset") || "0";
  const offset = Number(offsetRaw);
  const defaultDate = isNaN(offset) ? getDateStr(0) : getDateStr(offset);

  // 生クエリ取得
  const rawClass = c.req.query("class") || "";
  const rawType = c.req.query("type") || "";
  const rawDateRaw = c.req.query("date") || defaultDate;
  const rawPeriod = c.req.query("period") || "";
  const rawSubject = c.req.query("subject") || "";

  // 正規化関数: URLデコード→全角→半角→空白正規化
  const norm = (s: string) => normalizeWhitespace(fullwidthToHalfwidth(s));

  // フィルタ値生成
  const filterClass = rawClass ? norm(rawClass) : null;
  const filterType = rawType ? norm(rawType) : null;
  const filterDate = rawDateRaw ? norm(rawDateRaw) : null;
  const filterPeriod = rawPeriod ? norm(rawPeriod) : null;
  const filterSubj = rawSubject ? norm(rawSubject) : null;

  // HTML取得＆解析
  const url = "https://www.ibaraki-ct.ac.jp/info/archives/65544";
  const resp = await fetch(url);
  const html = await resp.text();
  const $ = load(html);
  const $main = $("#post_main");
  if (!$main.length) {
    return c.json({ error: "post_main が見つかりません" }, 404);
  }

  const results: Record<string, Entry[]> = {};
  let currentDate = "";

  $main.find("p").each((_, el) => {
    const $p = $(el);
    const text = norm($p.text());

    // 日付行
    if ($p.find("mark").length) {
      currentDate = text.replace(/\s+/g, "");
      results[currentDate] = [];
      return;
    }
    if (!currentDate) return;

    const m = entryRegex.exec(text);
    if (!m?.groups) return;
    const { symbol, cls, period, rest } = m.groups;
    const typeField = symbolMap(symbol);
    const classText = norm(cls);

    let subject1: string | null = null;
    let subject2: string | null = null;
    let period2: string | null = null;

    if (rest.includes("⇒")) {
      const [left, right] = rest.split("⇒", 2).map(norm);
      subject1 = left === "授業なし" ? null : left;
      const pr = periodRegex.exec(right);
      if (pr?.groups?.period2) {
        period2 = pr.groups.period2;
      } else {
        subject2 = right === "授業なし" ? null : right;
      }
    } else {
      subject1 = rest === "授業なし" ? null : norm(rest);
    }

    if (!period2) period2 = period;
    if (subject2 === null && period !== period2) {
      subject2 = subject1;
    }

    results[currentDate].push({
      type: typeField,
      class: classText,
      period1: period,
      period2,
      subject1,
      subject2,
    });
  });

  // フィルタ＆整形
  const data = Object.entries(results)
    .map(([date, entries]) => ({ date, entries }))
    .filter((g) => g.entries.length)
    .filter((g) => !filterDate || g.date === filterDate)
    .map((g) => ({
      date: g.date,
      entries: g.entries
        .filter((e) => !filterClass || e.class === filterClass)
        .filter((e) => !filterType || e.type === filterType)
        .filter(
          (e) =>
            !filterPeriod ||
            e.period1 === filterPeriod ||
            e.period2 === filterPeriod
        )
        .filter(
          (e) =>
            !filterSubj ||
            e.subject1?.includes(filterSubj) ||
            e.subject2?.includes(filterSubj)
        ),
    }))
    .filter((g) => g.entries.length);

  return c.json({ 休講情報: data });
});

export const GET = app.fetch;
