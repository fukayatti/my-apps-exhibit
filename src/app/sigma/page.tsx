// src/app/page.tsx
"use client";

import { useState, useEffect } from "react";
import { parse } from "mathjs";
import { BlockMath } from "react-katex";

export default function SigmaCalculatorPage() {
  const [from, setFrom] = useState<number>(1);
  const [to, setTo] = useState<number>(10);
  const [expression, setExpression] = useState<string>("k");
  const [result, setResult] = useState<number>(0);
  const [error, setError] = useState<string>("");
  const [latex, setLatex] = useState<string>("");

  useEffect(() => {
    try {
      setError("");
      const node = parse(expression);
      let sum = 0;
      for (let k = from; k <= to; k++) {
        const val = node.evaluate({ k });
        if (typeof val !== "number" || !isFinite(val)) {
          throw new Error(`k=${k} の評価が無効です: ${val}`);
        }
        sum += val;
      }
      setResult(sum);
      setLatex(`\\displaystyle \\sum_{k=${from}}^{${to}} ${expression}`);
    } catch (e: any) {
      setError(e.message);
      setResult(0);
      setLatex("");
    }
  }, [from, to, expression]);

  return (
    <div className="flex items-center justify-center py-12 px-4">
      <div className="w-full max-w-md bg-white shadow-lg rounded-xl p-6">
        <h1 className="text-2xl font-semibold text-gray-800 mb-6 text-center">
          Σ 計算機
        </h1>

        <label className="block mb-4">
          <span className="text-gray-700">下限 (k の開始値)</span>
          <input
            type="number"
            value={from}
            onChange={(e) => setFrom(parseInt(e.target.value) || 0)}
            className="mt-1 w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </label>

        <label className="block mb-4">
          <span className="text-gray-700">上限 (k の終了値)</span>
          <input
            type="number"
            value={to}
            onChange={(e) => setTo(parseInt(e.target.value) || 0)}
            className="mt-1 w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </label>

        <label className="block mb-6">
          <span className="text-gray-700">項の式 (例: k, k^2+3)</span>
          <input
            type="text"
            value={expression}
            onChange={(e) => setExpression(e.target.value)}
            className="mt-1 w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </label>

        <div className="mt-6 text-center">
          {error ? (
            <p className="text-red-600">{error}</p>
          ) : (
            <>
              <BlockMath math={latex} />
              <p className="mt-4 text-xl font-bold text-gray-800">= {result}</p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
