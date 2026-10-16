import useApi from "./useApi";
import { getCases } from "../api/cases";

export default function useCases() {
  const result = useApi(getCases);
  return { ...result, cases: result.data?.cases || (Array.isArray(result.data) ? result.data : []) };
}
