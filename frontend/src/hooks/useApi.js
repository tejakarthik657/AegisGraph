import { useCallback, useEffect, useState } from "react";

export default function useApi(apiFn, deps = [], immediate = true) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(immediate);
  const [error, setError] = useState(null);
  const execute = useCallback(async (...args) => {
    setLoading(true); setError(null);
    try { const result = await apiFn(...args); setData(result); return result; }
    catch (err) { const message = err?.response?.data?.detail || err?.message || "Request failed"; setError(message); return null; }
    finally { setLoading(false); }
  }, deps);
  useEffect(() => { if (immediate) execute(); }, [execute, immediate]);
  return { data, loading, error, execute, refetch: execute };
}
