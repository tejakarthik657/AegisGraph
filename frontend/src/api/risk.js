import client from "./client";

export const getRiskSummary = async () =>
  (await client.get("/risk-summary")).data;
export const getRisk = async (id) => (await client.get(`/risk/${id}`)).data;
