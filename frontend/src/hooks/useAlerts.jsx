import useApi from "./useApi";
import { getAlerts, getAlert } from "../services/api";

/**
 * Hook for fetching all alerts.
 */
export function useAlerts() {
  const { data, loading, error, refetch } = useApi(getAlerts);

  return {
    alerts: data?.alerts ?? [],
    totalAlerts: data?.total_alerts ?? 0,
    loading,
    error,
    refetch,
  };
}

/**
 * Hook for fetching a single alert by ID.
 */
export function useAlertDetail(alertId) {
  const fetcher = () => (alertId ? getAlert(alertId) : Promise.resolve(null));
  const { data, loading, error, refetch } = useApi(fetcher, [alertId]);

  return {
    alert: data,
    loading,
    error,
    refetch,
  };
}

export default useAlerts;
