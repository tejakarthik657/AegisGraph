import useApi from "./useApi";
import { getCases, getCase } from "../services/api";

/**
 * Hook for fetching all cases.
 */
export function useCases() {
  const { data, loading, error, refetch } = useApi(getCases);

  return {
    cases: data?.cases ?? [],
    totalCases: data?.total_cases ?? 0,
    loading,
    error,
    refetch,
  };
}

/**
 * Hook for fetching a single case by ID.
 */
export function useCaseDetail(caseId) {
  const fetcher = () => (caseId ? getCase(caseId) : Promise.resolve(null));
  const { data, loading, error, refetch } = useApi(fetcher, [caseId]);

  return {
    caseData: data,
    loading,
    error,
    refetch,
  };
}

export default useCases;
