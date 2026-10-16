import { useState, useEffect, useCallback } from "react";

/**
 * Generic hook for API calls with loading/error states.
 *
 * @param {Function} apiFn  — async function to call
 * @param {Array}    deps   — dependency array (triggers refetch)
 * @param {boolean}  immediate — fetch on mount (default true)
 */
export default function useApi(apiFn, deps = [], immediate = true) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(immediate);
  const [error, setError] = useState(null);

  const execute = useCallback(async (...args) => {
    setLoading(true);
    setError(null);

    try {
      const result = await apiFn(...args);
      setData(result);
      return result;
    } catch (err) {
      const message =
        err?.response?.data?.detail ||
        err?.message ||
        "An unexpected error occurred.";
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, deps);

  useEffect(() => {
    if (immediate) {
      execute();
    }
  }, [execute, immediate]);

  const refetch = useCallback(() => execute(), [execute]);

  return { data, loading, error, refetch, execute };
}
