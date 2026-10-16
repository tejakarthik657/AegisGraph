import { useMemo } from "react";
import useApi from "./useApi";
import { getTransactions } from "../services/api";

/**
 * Hook for paginated, filtered transaction data.
 */
export default function useTransactions(params = {}) {
  const { page = 1, pageSize = 25, search = "", typology = "", status = "", fraudOnly = false } = params;

  const fetcher = useMemo(
    () => () =>
      getTransactions({
        page,
        page_size: pageSize,
        search,
        typology,
        status,
        fraud_only: fraudOnly,
      }),
    [page, pageSize, search, typology, status, fraudOnly]
  );

  const { data, loading, error, refetch } = useApi(fetcher, [
    page,
    pageSize,
    search,
    typology,
    status,
    fraudOnly,
  ]);

  return {
    transactions: data?.transactions ?? [],
    pagination: data?.pagination ?? { page: 1, total: 0, total_pages: 0 },
    filters: data?.filters ?? {},
    loading,
    error,
    refetch,
  };
}
