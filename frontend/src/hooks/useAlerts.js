import useApi from "./useApi";
import { getAlerts } from "../api/alerts";

export default function useAlerts(params = {}) {
  const result = useApi(() => getAlerts(params), [JSON.stringify(params)]);
  return { ...result, alerts: result.data?.alerts || (Array.isArray(result.data) ? result.data : []) };
}
