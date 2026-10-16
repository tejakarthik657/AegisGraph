import { useMemo } from "react";
import useApi from "./useApi";
import { getTransactions } from "../api/transactions";

export default function useTransactions(params = {}) {
  const fetcher = useMemo(() => () => getTransactions(params), [params]);
  const result = useApi(fetcher, [fetcher]);
  return { ...result, transactions: result.data?.transactions || [], pagination: result.data?.pagination || {} };
}
